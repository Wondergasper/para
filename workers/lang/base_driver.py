from abc import ABC, abstractmethod
from workers.spec_generator.verifier import VerificationResult

class LanguageDriver(ABC):
    @abstractmethod
    def detect_func_name(self, source_code: str, default_name: str = "func") -> str:
        """Extract target function/subroutine name."""
        pass

    @abstractmethod
    def rename_main(self, source_code: str) -> str:
        """Rename main entry point/program block to avoid name collisions."""
        pass

    @abstractmethod
    def extract_functions(self, source_code: str) -> list[dict]:
        """Extract all function/subroutine definitions from source code."""
        pass

    @abstractmethod
    def analyze_loops(self, func_body: str) -> dict:
        """Scan loops for parallelization safety (no premature exits, no console IO)."""
        pass

    @abstractmethod
    def get_t1_system_prompt(self) -> str:
        """System prompt for Layer 1 classification."""
        pass

    @abstractmethod
    def get_t1_prompt(self, source_code: str) -> str:
        """User prompt for Layer 1 classification."""
        pass

    @abstractmethod
    def get_t2_system_prompt(self, target: str) -> str:
        """System prompt for Layer 2 candidate generation."""
        pass

    @abstractmethod
    def get_t2_prompt_hint(self, annotated_ir: dict) -> str:
        """Extra hints/instructions for Layer 2 based on analysis."""
        pass

    @abstractmethod
    def verify_candidate(
        self,
        candidate_code: str,
        reference_code: str,
        func_name: str,
        test_inputs: list,
        cfg,
    ) -> VerificationResult:
        """Verify compilation, correctness, and race-freedom of candidate code."""
        pass
