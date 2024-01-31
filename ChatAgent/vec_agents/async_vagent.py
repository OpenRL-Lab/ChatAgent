#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
import sys
from typing import Iterable, Callable, Optional, Union
from enum import Enum
import time
import multiprocessing as mp

from multiprocessing import Queue
from multiprocessing.connection import Connection

from gymnasium.vector.utils import CloudpickleWrapper, clear_mpi_env_vars
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium import logger

from ChatAgent.agents.base_agent import BaseAgent
from ChatAgent.vec_agents.base_vagent import BaseVecAgent


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_ACT = "act"


class AsyncVecAgent(BaseVecAgent):
    def __init__(self,
                 agent_fns: Iterable[Callable[[], BaseAgent]],
                 copy: bool = True,
                 context: Optional[str] = None,
                 daemon: bool = True,
                 worker: Optional[Callable] = None
                 ):
        ctx = mp.get_context(context)
        self.agent_fns = agent_fns
        self.copy = copy
        super().__init__(
            parallel_agent_num=len(agent_fns),
        )
        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = worker or _worker
        with clear_mpi_env_vars():
            for idx, agent_fn in enumerate(self.agent_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(agent_fn),
                        child_pipe,
                        parent_pipe,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()
        self._state = AsyncState.DEFAULT

    def _act(self, observations):
        self.act_send(observations)
        return self.act_fetch()

    def act_send(self, observations):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `act_send` while waiting for a pending call to"
                f" `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe, observation in zip(self.parent_pipes, observations):
            pipe.send(("act", observation))
        self._state = AsyncState.WAITING_ACT

    def act_fetch(self, timeout: Optional[Union[int, float]] = None):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_ACT:
            raise NoAsyncCallError(
                "Calling `act_fetch` without any prior call to `act_send`.",
                AsyncState.WAITING_ACT.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `act_fetch` has timed out after {timeout} second(s)."
            )

        actions = []

        successes = []

        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()

            successes.append(success)
            if success:
                action = result
                actions.append(action)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return actions

    def close_extras(
            self, timeout: Optional[Union[int, float]] = None, terminate: bool = False
    ):
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    "Calling `close` while waiting for a pending call to"
                    f" `{self._state.value}` to complete."
                )
                function = getattr(self, f"{self._state.value}_fetch")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                f"Trying to operate on `{type(self).__name__}`, after a call to"
                " `close()`."
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.parallel_agent_num - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(
                f"Received the following error from Worker-{index}: {exctype.__name__}:"
                f" {value}"
            )
            logger.error(f"Shutting down Worker-{index}.")
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                logger.error("Raising the last exception back to the main process.")
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector agent is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)


def _worker(
        index: int,
        agent_fn: callable,
        pipe: Connection,
        parent_pipe: Optional[Connection],
        error_queue: Queue,
):
    agent = agent_fn()
    if parent_pipe is not None:
        parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()
            if command == "act":
                action = agent.act(data)

                pipe.send((action, True))

            elif command == "close":
                pipe.send((None, True))
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`act`, `close`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        agent.close()
