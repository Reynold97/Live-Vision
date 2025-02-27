# src/core/state_machine.py
from enum import Enum, auto
import time
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Callable, Set
import logging
from datetime import datetime
from pydantic import BaseModel, Field

class PipelineState(str, Enum):
    """
    Pipeline states using explicit state machine pattern.
    Inherits from str to make it JSON serializable.
    """
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused" 
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"

class StateChangeEvent(BaseModel):
    """Event details when state changes."""
    previous_state: Optional[PipelineState] = None
    new_state: PipelineState
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PipelineStateMachine:
    """
    Manages state transitions for a pipeline with event hooks.
    """
    def __init__(self, pipeline_id: str, logger: Optional[logging.Logger] = None):
        self.pipeline_id = pipeline_id
        self.current_state = PipelineState.INITIALIZED
        self.state_history: List[StateChangeEvent] = []
        self.state_change_hooks: List[Callable[[StateChangeEvent], None]] = []
        self.state_start_time = time.time()
        self.logger = logger or logging.getLogger(__name__)
        self.lock = asyncio.Lock()
        
        # Record initial state
        self._record_state_change(None, self.current_state)
    
    def add_state_change_hook(self, hook: Callable[[StateChangeEvent], None]) -> None:
        """Add a hook to be called on state changes."""
        self.state_change_hooks.append(hook)
        
    def get_current_state(self) -> PipelineState:
        """Get the current pipeline state."""
        return self.current_state
        
    def get_state_duration(self) -> float:
        """Get duration in current state (seconds)."""
        return time.time() - self.state_start_time
        
    def can_transition_to(self, new_state: PipelineState) -> bool:
        """Check if a transition to new_state is valid from current state."""
        # Define valid state transitions
        valid_transitions = {
            PipelineState.INITIALIZED: {PipelineState.STARTING},
            PipelineState.STARTING: {PipelineState.RUNNING, PipelineState.FAILED},
            PipelineState.RUNNING: {PipelineState.PAUSING, PipelineState.STOPPING, PipelineState.COMPLETED, PipelineState.FAILED},
            PipelineState.PAUSING: {PipelineState.PAUSED, PipelineState.STOPPING, PipelineState.FAILED},
            PipelineState.PAUSED: {PipelineState.STARTING, PipelineState.STOPPING},
            PipelineState.STOPPING: {PipelineState.STOPPED, PipelineState.FAILED},
            PipelineState.STOPPED: {PipelineState.STARTING},
            PipelineState.FAILED: {PipelineState.STARTING},
            PipelineState.COMPLETED: {PipelineState.STARTING},
        }
        
        return new_state in valid_transitions.get(self.current_state, set())
    
    async def transition_to(self, new_state: PipelineState, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            new_state: The target state
            metadata: Optional metadata about the transition
            
        Returns:
            bool: True if transition succeeded, False otherwise
        """
        async with self.lock:
            if not self.can_transition_to(new_state):
                self.logger.warning(
                    f"Invalid state transition: {self.current_state} -> {new_state} for pipeline {self.pipeline_id}"
                )
                return False
                
            previous_state = self.current_state
            self.current_state = new_state
            self.state_start_time = time.time()
            
            # Record state change
            self._record_state_change(previous_state, new_state, metadata)
            
            return True
        
    def _record_state_change(self, previous_state: Optional[PipelineState], 
                            new_state: PipelineState,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a state change in history and trigger hooks."""
        timestamp = time.time()
        
        # Create event
        event = StateChangeEvent(
            previous_state=previous_state,
            new_state=new_state,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        # Add to history
        self.state_history.append(event)
        
        # Log the transition
        transition_str = f"{previous_state} -> {new_state}" if previous_state else f"Initial state: {new_state}"
        self.logger.info(f"Pipeline {self.pipeline_id} state change: {transition_str}")
        
        # Trigger hooks
        for hook in self.state_change_hooks:
            try:
                hook(event)
            except Exception as e:
                self.logger.error(f"Error in state change hook: {e}")
                
    def get_state_history(self) -> List[StateChangeEvent]:
        """Get the history of state changes."""
        return self.state_history.copy()
        
    def is_active(self) -> bool:
        """Check if the pipeline is in an active state."""
        return self.current_state in {
            PipelineState.STARTING,
            PipelineState.RUNNING,
            PipelineState.PAUSING
        }
        
    def is_terminal(self) -> bool:
        """Check if the pipeline is in a terminal state."""
        return self.current_state in {
            PipelineState.STOPPED,
            PipelineState.FAILED,
            PipelineState.COMPLETED
        }
        
    async def wait_for_state(self, target_state: PipelineState, timeout: Optional[float] = None) -> bool:
        """
        Wait asynchronously until pipeline reaches target state.
        
        Args:
            target_state: State to wait for
            timeout: Maximum seconds to wait or None for no timeout
            
        Returns:
            bool: True if state was reached, False if timed out
        """
        start_time = time.time()
        check_interval = 0.1  # seconds
        
        while True:
            if self.current_state == target_state:
                return True
                
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                return False
                
            await asyncio.sleep(check_interval)