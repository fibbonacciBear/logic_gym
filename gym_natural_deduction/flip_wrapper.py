
"""
Flip Wrapper
================
"""
from typing import Optional, Tuple
import pexpect
import sys

FLIP_HOME = "/Users/akashganesan/anaconda3/envs/drlzh/lib/python3.11/site-packages/flip"

# set sys.path to include the directory containing the FLiP module
sys.path.append(f"{FLIP_HOME}")
sys.path.append(f"{FLIP_HOME}/logic")
sys.path.append(f"{FLIP_HOME}/poset")


class FlipWrapper:
    """
    Wrapper for FliP.
    """

    def __init__(
        self, binary_path: str = "python"
    ):
        """
        Initialize the wrapper.
        """
        self.binary_path = binary_path
        self._proc = None
        self.command_line_arguments = (
            " -i -m flip.logic.fol_session"
        )

    def _get_stdout(self):
        self.proc.expect(
            [">>> ", "\.\.\. ",pexpect.EOF]
        )
        lines = self.proc.before.decode()
            # print(f"output:--{lines}--")
        if lines.startswith("Traceback"):
            raise Exception("Error occurred during execution: \n" + lines)
            
        return lines

    def start(
        self, list_of_fol_statements
    ):
        """
            foobar
        """
        if self._proc is not None:
            self._proc.close()

        self._proc = pexpect.spawn(
            f"{self.binary_path} {self.command_line_arguments} ",
            echo=False,
        )
        # https://pexpect.readthedocs.io/en/stable/commonissues.html#timing-issue-with-send-and-sendline
        self._proc.delaybeforesend = None  # type: ignore
        self._get_stdout()
        
        # catch the exception and throw an error
        for line in list_of_fol_statements + ["state()"]:
            self.proc.sendline(line)
            # print(f"input:--{line}--")
            output = self._get_stdout()
        
        return output.replace('\r', '')

    def run_proof_step(
        self, clause_label: str
    ) -> Tuple[Tuple[str, str, str], ...]:
        """
        Select a clause and get response from Vampire.

        :param clause_label: a given clause order number
        :returns: a sequence of action type, clause number and clause
        """
        self.proc.sendline(clause_label)
        return self._get_stdout()

    @property
    def proc(self) -> pexpect.spawn:
        """
        Return the pexpect.spawn object.
        """
        if self._proc is None:
            raise ValueError("start solving a problem first!")
        return self._proc

    def terminate(self) -> None:
        """Terminate Vampire process if any."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
