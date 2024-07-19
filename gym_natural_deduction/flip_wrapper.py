"""
Flip Wrapper
================
"""

from typing import Optional, Tuple, List
import pexpect
import sys
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

FLIP_HOME = (os.getenv("FLIP_HOME"),)

# set sys.path to include the directory containing the FLiP module
sys.path.append(f"{FLIP_HOME}")
sys.path.append(f"{FLIP_HOME}/logic")
sys.path.append(f"{FLIP_HOME}/poset")


class FlipWrapper:
    """
    Wrapper for FliP.
    """

    def __init__(self, binary_path: str = "python"):
        """
        Initialize the wrapper.
        """
        self.binary_path = binary_path
        self._proc = None
        self.command_line_arguments = " -i -m flip.logic.fol_session"

    def _get_stdout(self) -> str:
        """
        Retrieves the standard output from the process.

        Returns:
            str: The standard output from the process.

        Raises:
            Exception: If an error occurred during execution.
        """
        self.proc.expect([">>> ", "\.\.\. ", pexpect.EOF])
        lines = self.proc.before.decode()
        # print(f"output:--{lines}--")
        if lines.startswith("Traceback"):
            raise Exception("Error occurred during execution: \n" + lines)
        return lines

    def start(self, list_of_fol_statements: List[str]) -> str:
        """
        Starts the FLiP process and sends a list of FOL statements to it.

        Args:
            list_of_fol_statements (List[str]): A list of FOL statements to be sent to the FLiP process.

        Returns:
            str: The output received from the FLiP process.

        Raises:
            ValueError: If the FLiP process is already running.
        """
        if self._proc is not None:
            self._proc.close()

        # Start the FLiP process
        self._proc = pexpect.spawn(
            f"{self.binary_path} {self.command_line_arguments} ",
            echo=False,
        )
        self._proc.delaybeforesend = None  # Disable delay before sending

        # Get the initial output from the FLiP process
        self._get_stdout()

        # Send each FOL statement to the FLiP process and get the output
        for line in list_of_fol_statements + ["state()"]:
            self.proc.sendline(line)
            output = self._get_stdout()

        return output.replace("\r", "")

    def _get_goal_state(self, state: str) -> str:
        """
        Retrieves the goal_state from the given state string.

        Args:
            state (str): The state string containing the goal state.

        Returns:
            str: The goal state, which can be one of the following values:
                - "true" if the last state is equal to the goal state.
                - "false" if the last state is equal to "F".
                - "unknown" if the last state is neither equal to the goal state nor "F".
        """
        lines = state.split("\n")
        goal_line = ""

        for line in lines:
            line = line.strip()
            if line.endswith("Goal"):
                goal_line = line
                break

        goal = goal_line.split(" ")[0]
        lines = [line for line in lines if line != ""]
        last_state = lines[-1].split(" ")[0]

        if last_state == goal:
            return "true"
        elif last_state == "F":
            return "false"
        else:
            return "unknown"

    def run_proof_step(self, fol_statement: str) -> Tuple[str, str]:
        """
        Runs whatever is given through the FLiP process, and returns the new state, and a string indicating
        if the goal has been proved or disproved or is still unknown.
        """
        self.proc.sendline(fol_statement)
        self._get_stdout()

        self.proc.sendline("state()")
        state = self._get_stdout()

        self.proc.sendline("pp()")
        pp_state = self._get_stdout()

        goal_state = self._get_goal_state(pp_state)

        return goal_state, state.replace("\r", "")

    @property
    def proc(self) -> pexpect.spawn:
        """
        Return the pexpect.spawn object.
        """
        if self._proc is None:
            raise ValueError("Call start method first!")
        return self._proc

    def terminate(self) -> None:
        """Terminate FLiP process if any."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
