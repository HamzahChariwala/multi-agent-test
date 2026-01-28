"""Main orchestrator service entry point."""

import asyncio
import logging
import sys
from pathlib import Path

from orchestrator.config import OrchestratorConfig
from orchestrator.client import ModelClient, check_all_endpoints
from orchestrator.council_workflow import CouncilWorkflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('orchestrator.log'),
    ]
)
logger = logging.getLogger(__name__)


async def wait_for_services(
    client: ModelClient,
    config: OrchestratorConfig,
    max_wait: int = 300,
    check_interval: int = 5,
):
    """
    Wait for all services to be ready.
    
    Args:
        client: Model client
        config: Configuration
        max_wait: Maximum wait time in seconds
        check_interval: Check interval in seconds
    """
    logger.info("Waiting for services to be ready...")
    
    all_urls = config.get_all_member_urls() + [config.get_chairman_url()]
    
    elapsed = 0
    while elapsed < max_wait:
        health_map = await check_all_endpoints(client, all_urls)
        
        healthy_count = sum(1 for v in health_map.values() if v)
        total_count = len(all_urls)
        
        logger.info(f"Health check: {healthy_count}/{total_count} services ready")
        
        if healthy_count == total_count:
            logger.info("All services ready!")
            return True
        
        # Show which services are not ready
        for url, is_healthy in health_map.items():
            if not is_healthy:
                logger.warning(f"  {url}: NOT READY")
        
        await asyncio.sleep(check_interval)
        elapsed += check_interval
    
    logger.error(f"Timeout waiting for services after {max_wait}s")
    return False


async def run_example_task():
    """Run an example council task."""
    logger.info("="*80)
    logger.info("Running example council task")
    logger.info("="*80)
    
    # Load configuration
    config = OrchestratorConfig()
    logger.info(f"Loaded configuration: {config}")
    
    # Create client
    async with ModelClient(timeout=180) as client:
        # Wait for services
        ready = await wait_for_services(client, config, max_wait=60)
        
        if not ready:
            logger.error("Services not ready, exiting")
            return
        
        # Create workflow
        workflow = CouncilWorkflow(config, client)
        
        # Run council on example task
        task_prompt = """
        Design a scalable architecture for a real-time collaborative document editing system 
        (like Google Docs). Consider:
        1. How to handle concurrent edits from multiple users
        2. Conflict resolution strategies
        3. Data consistency and synchronization
        4. Performance optimization
        """
        
        logger.info(f"Task: {task_prompt.strip()}")
        
        result = await workflow.run(
            task_prompt=task_prompt,
            max_tokens=512,
            judging_rubric="Evaluate based on scalability, correctness, completeness, and feasibility.",
        )
        
        # Display results
        logger.info("="*80)
        logger.info("COUNCIL RESULTS")
        logger.info("="*80)
        logger.info(f"Stage: {result.stage.value}")
        logger.info(f"Request ID: {result.request_id}")
        
        logger.info(f"\nGeneration Phase:")
        logger.info(f"  Successes: {len(result.generation_outputs)}")
        logger.info(f"  Failures: {len(result.generation_failures)}")
        
        for url, output in result.generation_outputs.items():
            if output:
                logger.info(f"\n  {output.member_id}:")
                logger.info(f"    Answer: {output.answer[:200]}...")
                logger.info(f"    Confidence: {output.confidence}")
        
        logger.info(f"\nJudging Phase:")
        logger.info(f"  Total judgments: {len(result.judging_outputs)}")
        
        for i, judgment in enumerate(result.judging_outputs, 1):
            logger.info(f"\n  Judge {i} ({judgment.judge_id}):")
            logger.info(f"    Ranking: {judgment.ranking[:3]}")
            logger.info(f"    Top scores: {list(judgment.scores.items())[:3]}")
        
        logger.info(f"\nChairman Synthesis:")
        if result.final_output:
            logger.info(f"  Decision Trace: {result.final_output.decision_trace}")
            logger.info(f"  Selected Candidates: {result.final_output.selected_candidate_ids}")
            logger.info(f"  Confidence: {result.final_output.confidence}")
            logger.info(f"\n  Final Answer:")
            logger.info(f"  {result.final_output.final_answer[:500]}...")
        else:
            logger.error("  Chairman synthesis failed")
        
        if result.error:
            logger.error(f"\nError: {result.error}")
        
        logger.info("="*80)


async def interactive_mode():
    """Run orchestrator in interactive mode."""
    logger.info("Starting orchestrator in interactive mode")
    
    # Load configuration
    config = OrchestratorConfig()
    
    # Create client
    async with ModelClient(timeout=180) as client:
        # Wait for services
        ready = await wait_for_services(client, config)
        
        if not ready:
            logger.error("Services not ready, exiting")
            return
        
        # Create workflow
        workflow = CouncilWorkflow(config, client)
        
        logger.info("\n" + "="*80)
        logger.info("Multi-Agent Council Orchestrator - Interactive Mode")
        logger.info("="*80)
        logger.info("Enter tasks for the council to solve (Ctrl+C to exit)\n")
        
        try:
            while True:
                task = input("\nEnter task: ").strip()
                
                if not task:
                    continue
                
                logger.info(f"\nProcessing task: {task}")
                
                result = await workflow.run(task_prompt=task, max_tokens=512)
                
                print("\n" + "="*80)
                print("RESULTS")
                print("="*80)
                
                if result.final_output:
                    print(f"\nFinal Answer:")
                    print(result.final_output.final_answer)
                    print(f"\nConfidence: {result.final_output.confidence}")
                    print(f"Based on: {', '.join(result.final_output.selected_candidate_ids)}")
                else:
                    print("ERROR: Council failed to produce answer")
                    if result.error:
                        print(f"Error: {result.error}")
                
                print("="*80)
        
        except KeyboardInterrupt:
            logger.info("\nShutting down...")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Council Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["example", "interactive"],
        default="example",
        help="Run mode: example task or interactive",
    )
    
    args = parser.parse_args()
    
    if args.mode == "example":
        asyncio.run(run_example_task())
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()

