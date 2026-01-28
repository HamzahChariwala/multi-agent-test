"""Integration tests for the multi-agent council system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from schemas.generation import GenerationRequest, GenerationOutput
from schemas.judging import JudgingRequest, JudgingOutput
from schemas.chairman import ChairmanRequest, ChairmanOutput
from orchestrator.client import ModelClient
from orchestrator.council_workflow import CouncilWorkflow, WorkflowStage
from orchestrator.config import OrchestratorConfig


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    with patch('orchestrator.config.Path.open'):
        config = MagicMock(spec=OrchestratorConfig)
        config.get_all_member_urls.return_value = [
            "http://localhost:8001",
            "http://localhost:8002",
            "http://localhost:8003",
        ]
        config.get_chairman_url.return_value = "http://localhost:8020"
        return config


@pytest.fixture
def mock_client():
    """Create mock client."""
    client = AsyncMock(spec=ModelClient)
    return client


@pytest.mark.asyncio
async def test_generation_phase(mock_config, mock_client):
    """Test generation phase."""
    # Setup mock responses
    mock_client.generate.side_effect = [
        GenerationOutput(
            answer="Answer 1",
            assumptions=["Assumption 1"],
            confidence=0.8,
            risks=[],
            member_id="member_1",
        ),
        GenerationOutput(
            answer="Answer 2",
            assumptions=[],
            confidence=0.9,
            risks=["Risk 1"],
            member_id="member_2",
        ),
        GenerationOutput(
            answer="Answer 3",
            assumptions=[],
            confidence=0.7,
            risks=[],
            member_id="member_3",
        ),
    ]
    
    # Create workflow
    workflow = CouncilWorkflow(mock_config, mock_client)
    
    # Run
    result = await workflow.run("Test task", max_tokens=100)
    
    # Verify
    assert result.stage == WorkflowStage.COMPLETE or result.stage == WorkflowStage.GENERATION_COMPLETE
    assert len(result.generation_outputs) > 0


@pytest.mark.asyncio
async def test_full_workflow(mock_config, mock_client):
    """Test complete workflow."""
    # Mock generation responses
    async def mock_generate(url, request):
        member_id = url.split(":")[-1]  # Use port as ID
        return GenerationOutput(
            answer=f"Answer from {member_id}",
            assumptions=[],
            confidence=0.8,
            risks=[],
            member_id=f"member_{member_id}",
        )
    
    # Mock judging responses
    async def mock_judge(url, request):
        judge_id = url.split(":")[-1]
        candidate_ids = list(request.candidates.keys())
        return JudgingOutput(
            scores={cid: 0.5 + (hash(cid) % 50) / 100.0 for cid in candidate_ids},
            ranking=sorted(candidate_ids, reverse=True),
            top_reasoning={candidate_ids[0]: "Best" if candidate_ids else ""},
            judge_id=f"judge_{judge_id}",
        )
    
    # Mock chairman response
    async def mock_synthesize(url, request):
        return ChairmanOutput(
            final_answer="Synthesized final answer",
            decision_trace="Used top candidates",
            selected_candidate_ids=list(request.candidates.keys())[:2],
            confidence=0.9,
        )
    
    mock_client.generate.side_effect = mock_generate
    mock_client.judge.side_effect = mock_judge
    mock_client.synthesize.side_effect = mock_synthesize
    
    # Create workflow
    workflow = CouncilWorkflow(mock_config, mock_client)
    
    # Run
    result = await workflow.run("Design a scalable system", max_tokens=512)
    
    # Verify
    assert result.stage == WorkflowStage.COMPLETE
    assert len(result.generation_outputs) > 0
    assert len(result.judging_outputs) > 0
    assert result.final_output is not None
    assert result.final_output.confidence > 0


@pytest.mark.asyncio
async def test_partial_failures(mock_config, mock_client):
    """Test workflow with some failures."""
    # Mock with some failures
    async def mock_generate_with_failures(url, request):
        if "8001" in url:
            return None  # Simulate failure
        member_id = url.split(":")[-1]
        return GenerationOutput(
            answer=f"Answer from {member_id}",
            assumptions=[],
            confidence=0.8,
            risks=[],
            member_id=f"member_{member_id}",
        )
    
    async def mock_judge(url, request):
        judge_id = url.split(":")[-1]
        candidate_ids = list(request.candidates.keys())
        return JudgingOutput(
            scores={cid: 0.7 for cid in candidate_ids},
            ranking=candidate_ids,
            top_reasoning={candidate_ids[0]: "Good" if candidate_ids else ""},
            judge_id=f"judge_{judge_id}",
        )
    
    async def mock_synthesize(url, request):
        return ChairmanOutput(
            final_answer="Final answer despite failures",
            decision_trace="Worked with available data",
            selected_candidate_ids=list(request.candidates.keys()),
            confidence=0.75,
        )
    
    mock_client.generate.side_effect = mock_generate_with_failures
    mock_client.judge.side_effect = mock_judge
    mock_client.synthesize.side_effect = mock_synthesize
    
    # Create workflow
    workflow = CouncilWorkflow(mock_config, mock_client)
    
    # Run
    result = await workflow.run("Test task", max_tokens=100)
    
    # Verify - should still complete with partial results
    assert result.stage == WorkflowStage.COMPLETE
    assert len(result.generation_failures) > 0
    assert len(result.generation_outputs) > 0  # Some succeeded


def test_schema_validation():
    """Test schema validation."""
    # Test GenerationOutput
    output = GenerationOutput(
        answer="Test answer",
        assumptions=["A1", "A2"],
        confidence=0.85,
        risks=["R1"],
        member_id="member_1",
    )
    
    assert output.answer == "Test answer"
    assert len(output.assumptions) == 2
    assert 0 <= output.confidence <= 1
    
    # Test JudgingOutput
    judging = JudgingOutput(
        scores={"m1": 0.9, "m2": 0.7},
        ranking=["m1", "m2"],
        top_reasoning={"m1": "Best solution"},
        judge_id="judge_1",
    )
    
    assert len(judging.scores) == 2
    assert judging.ranking[0] == "m1"
    
    # Test ChairmanOutput
    chairman = ChairmanOutput(
        final_answer="Final answer",
        decision_trace="Decision process",
        selected_candidate_ids=["m1", "m2"],
        confidence=0.9,
    )
    
    assert chairman.final_answer == "Final answer"
    assert len(chairman.selected_candidate_ids) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

