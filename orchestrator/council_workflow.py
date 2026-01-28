"""Council workflow state machine (Generate → Judge → Chairman)."""

import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from schemas.generation import GenerationRequest, GenerationOutput
from schemas.judging import JudgingRequest, JudgingOutput
from schemas.chairman import ChairmanRequest, ChairmanOutput

from orchestrator.client import ModelClient, parallel_generate, parallel_judge
from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Workflow stages."""
    INITIALIZED = "initialized"
    GENERATING = "generating"
    GENERATION_COMPLETE = "generation_complete"
    JUDGING = "judging"
    JUDGING_COMPLETE = "judging_complete"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class CouncilResult:
    """Complete council result."""
    request_id: str
    task_prompt: str
    stage: WorkflowStage
    
    # Generation phase
    generation_outputs: Dict[str, GenerationOutput] = field(default_factory=dict)
    generation_failures: List[str] = field(default_factory=list)
    
    # Judging phase
    judging_outputs: List[JudgingOutput] = field(default_factory=list)
    judging_failures: List[str] = field(default_factory=list)
    
    # Chairman phase
    final_output: Optional[ChairmanOutput] = None
    
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "task_prompt": self.task_prompt,
            "stage": self.stage.value,
            "generation_outputs": {
                url: output.model_dump() if output else None
                for url, output in self.generation_outputs.items()
            },
            "generation_failures": self.generation_failures,
            "judging_outputs": [
                output.model_dump() if output else None
                for output in self.judging_outputs
            ],
            "judging_failures": self.judging_failures,
            "final_output": self.final_output.model_dump() if self.final_output else None,
            "error": self.error,
        }


class CouncilWorkflow:
    """
    3-stage council workflow orchestrator.
    
    Stage 1: Generate - All members generate candidate answers
    Stage 2: Judge - All members judge the candidates
    Stage 3: Chairman - Chairman synthesizes final answer
    """
    
    def __init__(
        self,
        config: OrchestratorConfig,
        client: ModelClient,
    ):
        """
        Initialize workflow.
        
        Args:
            config: Orchestrator configuration
            client: Model client for HTTP requests
        """
        self.config = config
        self.client = client
    
    async def run(
        self,
        task_prompt: str,
        max_tokens: int = 512,
        judging_rubric: Optional[str] = None,
    ) -> CouncilResult:
        """
        Run complete council workflow.
        
        Args:
            task_prompt: The task to solve
            max_tokens: Maximum tokens for generation
            judging_rubric: Optional custom rubric for judging
        
        Returns:
            CouncilResult with all outputs
        """
        # Initialize result
        request_id = str(uuid.uuid4())
        result = CouncilResult(
            request_id=request_id,
            task_prompt=task_prompt,
            stage=WorkflowStage.INITIALIZED,
        )
        
        logger.info(f"[{request_id}] Starting council workflow for task")
        
        try:
            # Stage 1: Generate
            result.stage = WorkflowStage.GENERATING
            await self._run_generation(result, max_tokens)
            
            if not result.generation_outputs:
                result.stage = WorkflowStage.FAILED
                result.error = "All generation requests failed"
                return result
            
            result.stage = WorkflowStage.GENERATION_COMPLETE
            logger.info(
                f"[{request_id}] Generation complete: "
                f"{len(result.generation_outputs)} successes, "
                f"{len(result.generation_failures)} failures"
            )
            
            # Stage 2: Judge
            result.stage = WorkflowStage.JUDGING
            await self._run_judging(result, judging_rubric)
            
            if not result.judging_outputs:
                result.stage = WorkflowStage.FAILED
                result.error = "All judging requests failed"
                return result
            
            result.stage = WorkflowStage.JUDGING_COMPLETE
            logger.info(
                f"[{request_id}] Judging complete: "
                f"{len(result.judging_outputs)} judgments received"
            )
            
            # Stage 3: Chairman Synthesis
            result.stage = WorkflowStage.SYNTHESIZING
            await self._run_synthesis(result)
            
            if result.final_output is None:
                result.stage = WorkflowStage.FAILED
                result.error = "Chairman synthesis failed"
                return result
            
            result.stage = WorkflowStage.COMPLETE
            logger.info(f"[{request_id}] Council workflow complete")
        
        except Exception as e:
            logger.error(f"[{request_id}] Workflow error: {e}", exc_info=True)
            result.stage = WorkflowStage.FAILED
            result.error = str(e)
        
        return result
    
    async def _run_generation(self, result: CouncilResult, max_tokens: int):
        """Run generation phase."""
        # Create generation request
        gen_request = GenerationRequest(
            task_prompt=result.task_prompt,
            max_tokens=max_tokens,
            request_id=result.request_id,
        )
        
        # Send to all members in parallel
        member_urls = self.config.get_all_member_urls()
        outputs = await parallel_generate(self.client, member_urls, gen_request)
        
        # Process results
        for url, output in outputs.items():
            if output is not None:
                result.generation_outputs[url] = output
            else:
                result.generation_failures.append(url)
    
    async def _run_judging(self, result: CouncilResult, rubric: Optional[str]):
        """Run judging phase."""
        # Prepare candidates dictionary
        candidates = {
            output.member_id: output.answer
            for output in result.generation_outputs.values()
            if output and output.member_id
        }
        
        if not candidates:
            logger.error(f"[{result.request_id}] No valid candidates for judging")
            return
        
        # Create judging request
        judge_request = JudgingRequest(
            task_prompt=result.task_prompt,
            candidates=candidates,
            rubric=rubric or "Evaluate based on correctness, completeness, clarity, and feasibility.",
            request_id=result.request_id,
        )
        
        # Send to all members in parallel
        member_urls = self.config.get_all_member_urls()
        outputs = await parallel_judge(self.client, member_urls, judge_request)
        
        # Process results
        for url, output in outputs.items():
            if output is not None:
                result.judging_outputs.append(output)
            else:
                result.judging_failures.append(url)
    
    async def _run_synthesis(self, result: CouncilResult):
        """Run chairman synthesis phase."""
        # Prepare candidates dictionary
        candidates = {
            output.member_id: output.answer
            for output in result.generation_outputs.values()
            if output and output.member_id
        }
        
        # Prepare judgments list
        judgments = [
            output.model_dump()
            for output in result.judging_outputs
        ]
        
        if not candidates or not judgments:
            logger.error(f"[{result.request_id}] Insufficient data for synthesis")
            return
        
        # Create chairman request
        chairman_request = ChairmanRequest(
            task_prompt=result.task_prompt,
            candidates=candidates,
            judgments=judgments,
            request_id=result.request_id,
        )
        
        # Send to chairman
        chairman_url = self.config.get_chairman_url()
        output = await self.client.synthesize(chairman_url, chairman_request)
        
        result.final_output = output


async def run_council(
    task_prompt: str,
    config_path: str = "./config/endpoints.yaml",
    max_tokens: int = 512,
    judging_rubric: Optional[str] = None,
) -> CouncilResult:
    """
    Convenience function to run council workflow.
    
    Args:
        task_prompt: Task to solve
        config_path: Path to configuration
        max_tokens: Maximum tokens for generation
        judging_rubric: Optional judging rubric
    
    Returns:
        CouncilResult
    """
    config = OrchestratorConfig(endpoints_config_path=config_path)
    
    async with ModelClient() as client:
        workflow = CouncilWorkflow(config, client)
        result = await workflow.run(task_prompt, max_tokens, judging_rubric)
    
    return result

