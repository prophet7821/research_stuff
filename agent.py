import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import generate_response_llama

class ModelWrapper(ABC):
    """Base abstract class for all model wrappers"""
    @abstractmethod
    def generate(self, prompt):
        """Generate a response for the given prompt"""
        pass

class DummyModel(ModelWrapper):
    """Simple echo model for testing purposes"""
    def generate(self, prompt):
        return prompt

class BaseAgent(ModelWrapper):
    """Single language model agent that can participate in debates"""

    def __init__(self, model_name, device="cuda", intention="neutral", is_chat=0):
        """
        Initialize a language model agent

        Args:
            model_name: HuggingFace model name/path
            device: Device to run model on ("cuda", "cuda:0", etc)
            intention: Agent behavior ("neutral", "harmful", or "harmless")
            is_chat: Whether model is a chat model (affects prompt template)
        """
        # Validate intention
        if intention not in ['neutral', 'harmful', 'harmless']:
            raise ValueError("intention must be one of [neutral, harmful, harmless]")

        self.intention = intention
        self.device = device

        # Load prompt templates based on intention and model type
        prompt_file = './prompts/nonchat_prompts.json'
        self.prompts = json.load(open(prompt_file, "r"))[intention]

        print(f"Using model {model_name} with intention {intention} on device {device}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set up generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=0.7,
            max_new_tokens=256,
            top_p=0.95,
            repetition_penalty=1.15,
            device=self.device,
        )

    def generate(self, context):
        """Generate a response given a context (prompt or message list)"""
        if isinstance(context, str):
            context = [self.construct_initial_message(context)]
        return generate_response_llama(self.pipeline, context)

    def construct_assistant_message(self, completion):
        """Format model output as an assistant message"""
        return {"role": "assistant", "content": completion}

    def construct_initial_message(self, prompt):
        """Format the initial user message with the prompt template"""
        return {"role": "user", "content": self.prompts["init"].replace("<TOPIC>", prompt)}

    def construct_discussion_message(self, topic, feedback_list):
        """Format a follow-up message with feedback from other agents"""
        if not feedback_list:
            # Self-reflection prompt if no feedback
            return {"role": "user", "content": self.prompts["self-reflect"].replace("<TOPIC>", topic)}

        # Format feedback from other agents
        formatted_feedback = " ".join([f"One agent response: ```{feedback}```" for feedback in feedback_list])

        # Create discussion prompt with feedback
        message = self.prompts["discussion"].replace("<TOPIC>", topic).replace("<FEEDBACK>", formatted_feedback)
        return {"role": "user", "content": message}

class DebateAgent(ModelWrapper):
    """Base class for debate between multiple agents"""

    def __init__(self, agents, n_discussion_rounds=2):
        """
        Initialize a debate among multiple agents

        Args:
            agents: List of BaseAgent instances
            n_discussion_rounds: Number of rounds of debate
        """
        self.agents = agents
        self.n_agents = len(agents)
        self.n_discussion_rounds = n_discussion_rounds

    def generate(self, initial_prompt):
        """Run a debate on the initial prompt and return all responses"""
        # Initialize conversation contexts for each agent
        agent_contexts = [[agent.construct_initial_message(initial_prompt)] for agent in self.agents]

        # Get initial responses from all agents
        for i, agent in enumerate(self.agents):
            response = agent.generate(agent_contexts[i])
            agent_contexts[i].append(agent.construct_assistant_message(response))

        # Run debate rounds
        for _ in range(self.n_discussion_rounds):
            for i, agent in enumerate(self.agents):
                # Get feedback from other agents
                other_agents_feedback = [
                    context[-1]["content"]
                    for idx, context in enumerate(agent_contexts)
                    if idx != i
                ]

                # Create discussion message with feedback
                discussion_msg = agent.construct_discussion_message(initial_prompt, other_agents_feedback)
                agent_contexts[i].append(discussion_msg)

                # Generate updated response
                response = agent.generate(agent_contexts[i])
                agent_contexts[i].append(agent.construct_assistant_message(response))

        # Format results: extract all agent responses by round
        results = []
        for i in range(self.n_agents):
            agent_responses = [msg["content"] for msg in agent_contexts[i][1::2]]
            results.append(agent_responses)

        return results

class NonDiverseDebateAgent(DebateAgent):
    """Debate among multiple instances of the same model"""

    def __init__(self, model_name, n_agents=2, n_discussion_rounds=2):
        """
        Initialize debate with multiple copies of the same model

        Args:
            model_name: Model name/path or list of model names
            n_agents: Number of agent instances to create
            n_discussion_rounds: Number of rounds of debate
        """
        # Support either a single model name or a list of model names
        if isinstance(model_name, str):
            model_name = [model_name] * n_agents

        # Create agents, distributing across available GPUs
        agents = [
            BaseAgent(model_name[0], device="cuda:0", intention="harmful"),
            BaseAgent(model_name[1], device="cuda:1", intention="harmless")
        ]

        super().__init__(agents, n_discussion_rounds)

class DiverseDebateAgent(DebateAgent):
    """Debate among different pre-initialized models"""
    # This is just an alias for the base DebateAgent class
    pass