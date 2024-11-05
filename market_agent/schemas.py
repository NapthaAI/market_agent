from pydantic import BaseModel
from market_agent.persona import Persona
from pydantic_settings.main import SettingsConfigDict 
from pydantic.functional_validators import model_validator 
from typing import Literal 
from typing_extensions import Self 
from typing import Optional 
from pydantic.fields import Field 
from typing import Dict 
from typing import List 
from typing import Union 
from pydantic_settings.main import BaseSettings 
from abc import abstractmethod 
from abc import ABC 
from typing import Any 
from pydantic.fields import computed_field 
import random 
from typing import Tuple 
from typing import Type 

class LLMConfig(BaseModel):
    client: Literal["openai", "azure_openai", "anthropic", "vllm", "litellm"]
    model: Optional[str] = None
    max_tokens: int = Field(default=400)
    temperature: float = 0
    response_format: Literal["json_beg", "text","json_object","structured_output","tool"] = "text"
    use_cache: bool = True

    @model_validator(mode="after")
    def validate_response_format(self) -> Self:
        if self.response_format == "json_object" and self.client in ["vllm", "litellm","anthropic"]:
            raise ValueError(f"{self.client} does not support json_object response format")
        elif self.response_format == "structured_output" and self.client == "anthropic":
            raise ValueError(f"Anthropic does not support structured_output response format use json_beg or tool instead")
        return self

class AgentConfig(BaseModel):
    num_units: int
    base_value: float
    use_llm: bool
    buyer_initial_cash: float
    buyer_initial_goods: int
    seller_initial_cash: float
    seller_initial_goods: int
    good_name: str
    noise_factor: float
    max_relative_spread: float

class GroupChatConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    initial_topic: str

class EnvironmentConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    good_name: str

class OrchestratorConfig(BaseSettings):
    num_agents: int
    max_rounds: int
    agent_config: AgentConfig
    llm_configs: List[LLMConfig]
    environment_configs: Dict[str, Union[EnvironmentConfig, GroupChatConfig]]
    protocol: str
    database_config: Dict[str, str]

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

class LocalAction(BaseModel, ABC):
    """Represents an action for a single agent."""
    agent_id: str
    action: Any

    @classmethod
    @abstractmethod
    def sample(cls, agent_id: str) -> 'LocalAction':
        """Sample a random action for the given agent_id."""
        pass

class GlobalAction(BaseModel):
    """Represents actions for all agents."""
    actions: Dict[str, LocalAction]

    def locals(self) -> Dict[str, LocalAction]:
        """Get the local actions for all agents."""
        return self.actions

    @classmethod
    def from_local_actions(cls, local_actions: Dict[str, LocalAction]) -> "GlobalAction":
        """Create a global action from local actions."""
        return cls(actions=local_actions)

class LocalObservation(BaseModel, ABC):
    """Represents an observation for a single agent."""
    agent_id: str
    observation: BaseModel

class GlobalObservation(BaseModel):
    """Represents observations for all agents."""
    observations: Dict[str, LocalObservation]
    

    def locals(self) -> Dict[str, LocalObservation]:
        """Get the local observations for all agents."""
        return self.observations
    
    @property
    @computed_field
    def global_obs(self) -> Optional[Any]:
        """Get the global observation for all agents."""
        return None

    @classmethod
    def from_local_observations(cls, local_observations: Dict[str, LocalObservation]) -> "GlobalObservation":
        """Create a global observation from local observations."""
        return cls(observations=local_observations)

    def to_local(self, agent_id: str) -> LocalObservation:
        """Convert global observation to local observation for a specific agent."""
        return self.observations[agent_id]

class LocalEnvironmentStep(BaseModel):
    """Represents the output of a single environment step for a single agent."""
    observation: LocalObservation
    done: bool
    info: Dict[str, Any]

class EnvironmentStep(BaseModel):
    """Represents the output of a single environment step."""
    global_observation: GlobalObservation
    done: bool
    info: Dict[str, Any]

    @classmethod
    def from_local_steps(cls, local_steps: Dict[str, LocalEnvironmentStep]) -> "EnvironmentStep":
        """Create a global environment step from local steps."""
        observations = {agent_id: step.observation for agent_id, step in local_steps.items()}
        done = all(step.done for step in local_steps.values())
        info = {}
        return cls(
            global_observation=GlobalObservation.from_local_observations(observations),
            done=done,
            info=info
        )

    def get_local_step(self, agent_id: str) -> LocalEnvironmentStep:
        """Get the local step for a single agent."""
        return LocalEnvironmentStep(
            observation=self.global_observation.to_local(agent_id),
            done=self.done,
            info=self.info
        )

class EnvironmentHistory(BaseModel):
    """Represents the history of environment steps."""
    steps: List[Tuple[GlobalAction, EnvironmentStep]] = Field(default_factory=list)

    def add_step(self, action: GlobalAction, step: EnvironmentStep):
        """Add a step to the history."""
        self.steps.append((action, step))

class ActionSpace(BaseModel):
    allowed_actions: List[Type[LocalAction]] = Field(default_factory=list, description="List of allowed action types")

    def sample(self, agent_id: str) -> LocalAction:
        """Sample a random action from the allowed actions."""
        if not self.allowed_actions:
            raise ValueError("No allowed actions defined")
        action_type = random.choice(self.allowed_actions)
        return action_type.sample(agent_id)
    
    def get_action_schema(self) -> Type[BaseModel]:
        """Get the schema for the allowed actions."""
        if not self.allowed_actions:
            raise ValueError("No allowed actions defined")
        # Assuming all allowed actions have the same schema
        return self.allowed_actions[0].action_schema()  

class ObservationSpace(BaseModel):
    allowed_observations: List[Type[LocalObservation]] = Field(default_factory=list)

class Mechanism(BaseModel, ABC):
    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")
    @abstractmethod
    def step(self, action: Union[LocalAction, GlobalAction]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Execute a step in the mechanism."""
        pass


    @abstractmethod
    def get_global_state(self) -> Any:
        """Get the global state of the mechanism."""
        pass

class StrObservation(LocalObservation):
    observation: str

    @classmethod
    def sample(cls, agent_id: str, min_length: int = 1, max_length: int = 100) -> 'StrObservation':
        content = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=random.randint(min_length, max_length)))
        return cls(agent_id=agent_id, observation=content)

class Notebook(Mechanism):
    text: str = Field(default="", description="The notebook's text content")
    
    sequential: bool = Field(default=True, description="Whether the mechanism is sequential")

    def step(self, action: LocalAction) -> LocalEnvironmentStep:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n[{timestamp}] Agent {action.agent_id}:"
        self.text += f"{header}\n{action.action}\n"
        
        observation = StrObservation(agent_id=action.agent_id, observation=self.text)
        done = False  # The notebook never ends
        info = {}

        return LocalEnvironmentStep(
            observation=observation,
            done=done,
            info=info
        )

    def get_global_state(self) -> str:
        return self.text

class StrAction(LocalAction):
    action: str = Field(..., description="Content of the string action")

    @classmethod
    def sample(cls, agent_id: str, min_length: int = 1, max_length: int = 10) -> 'StrAction':
        content = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(min_length, max_length)))
        return cls(agent_id=agent_id, action=content)

class NotebookActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [StrAction]

    def sample(self, agent_id: str) -> StrAction:
        content = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 20)))
        return StrAction(agent_id=agent_id, action=content)

class NotebookObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [StrObservation]

    def sample(self, agent_id: str) -> LocalObservation:
        return StrObservation.sample(agent_id)

class Protocol(BaseModel, ABC):
    @abstractmethod
    def parse_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_message(self, *args, **kwargs) -> 'Protocol':
        pass

    @classmethod
    @abstractmethod
    def create_message(cls, *args, **kwargs) -> 'Protocol':
        pass

class MultiAgentEnvironment(BaseModel):
    """
    Base class for multi-agent environments. With batched or sequential actions.
    """
    name: str = Field(..., description="Name of the environment")
    address: str = Field(..., description="Address of the environment for orchestrator linking")
    current_step: int = Field(default=0, description="Current step/round of the simulation")
    max_steps: int = Field(..., description="Maximum number of steps/rounds for this environment")
    action_space: ActionSpace = Field(default_factory=NotebookActionSpace, description="Action space of the environment")
    observation_space: ObservationSpace = Field(default_factory=NotebookObservationSpace, description="Observation space of the environment")
    history: EnvironmentHistory = Field(default_factory=EnvironmentHistory, description="History of environment steps")
    mechanism: Mechanism = Field(default_factory=Notebook, description="Mechanism of the environment that determines the rules of the game P(s, a, s')")

    def step(self, actions: GlobalAction) -> EnvironmentStep:
        """
        Run one timestep of the environment's dynamics using the batched agent actions.
        
        Args:
            actions (GlobalAction): A batched action containing actions for each agent.

        Returns:
            EnvironmentStep: The result of taking a step in the environment.
        """
        if self.mechanism.sequential:
            # if it is sequential, we need to run the mechanism for each agent
            local_steps: Dict[str, LocalEnvironmentStep] = {}  # Correct type annotation
            for agent_id, local_action in actions.locals().items():
                local_step = self.mechanism.step(local_action)
                assert isinstance(local_step, LocalEnvironmentStep)
                local_steps[agent_id] = local_step
            global_step = EnvironmentStep.from_local_steps(local_steps)
        else:
            global_step = self.mechanism.step(actions)
            assert isinstance(global_step, EnvironmentStep)
        self.current_step += 1
        self.update_history(actions, global_step)
        return global_step

    def reset(self) -> GlobalObservation:
        """
        Reset the environment and return the initial global observation.

        Returns:
            GlobalObservation: Initial global observation of the environment.
        """
        self.current_step = 0
        self.global_state = {}
        self.history = EnvironmentHistory()
        if isinstance(self.mechanism, Notebook):
            self.mechanism.text = ""
        return GlobalObservation(observations={})

    def render(self):
        """
        Render the environment.
        """
        print(self.get_global_state())

    def close(self):
        """
        Close the environment, do any necessary cleanup.
        """
        pass  # No specific cleanup needed for the basic environment

    def get_global_state(self) -> Any:
        """
        Return a summary of the global state.

        Returns:
            Any: The global state.
        """
        return self.mechanism.get_global_state()

    def get_current_step(self) -> int:
        """
        Return the current step/round of the simulation.

        Returns:
            int: The current step.
        """
        return self.current_step

    def update_history(self, action: GlobalAction, step: EnvironmentStep):
        """
        Update the environment history with the latest step.
        """
        self.history.add_step(action, step)

    def random_action_test(self, num_agents: int, num_steps: int):
        """
        Run a test with random actions for the specified number of agents and steps.
        """
        agent_ids = [f"Agent{i}" for i in range(num_agents)]

        print(f"\n=== Random Action Test for {self.name} ===\n")

        for step in range(num_steps):
            print(f"\nStep {step + 1}:")

            actions = {}
            for agent_id in agent_ids:
                action_type = random.choice(self.action_space.allowed_actions)
                actions[agent_id] = action_type.sample(agent_id)

            global_action = GlobalAction(actions=actions)
            step_result = self.step(global_action)
            self._print_step_results(step_result, actions, agent_ids)

        print("\nTest completed.")
        self.close()

    def _print_step_results(self, step_result: EnvironmentStep, actions: Dict[str, LocalAction], agent_ids: List[str]):
        """
        Print the results of a single step. This method can be overridden in subclasses for custom output.
        """
        for agent_id in agent_ids:
            local_step = step_result.get_local_step(agent_id)
            print(f"{agent_id} action: {actions[agent_id].action}")
            print(f"{agent_id} observation: {step_result.global_observation.observations[agent_id].observation}")

        print("\nGlobal state:")
        print(self.get_global_state())
        print("\n" + "="*50)

class InputSchema(BaseModel):
    tool_name: str
    tool_input_type: str
    tool_input_value: str
    agent_id: int
    persona: Persona
    llm_config: LLMConfig
    config: OrchestratorConfig
    environments: dict
