# madtf.py
import dspy
import os
from dspy.functional import TypedPredictor
from dotenv import load_dotenv
from transitions import Machine
import asyncio
from typing import List, Dict

load_dotenv()

# Create a language model using the Claude API
llm = dspy.OpenAI(
    model='gpt-4o',
    api_key=os.environ['OPENAI_API_KEY'],
    max_tokens=2000
)

dspy.settings.configure(lm=llm)

if not os.path.exists('scratchpad'):
    os.makedirs('scratchpad')

if not os.path.exists('versions'):
    os.makedirs('versions')

class TeamFormationSignature(dspy.Signature):
    task_description = dspy.InputField(desc="The description of the task")
    team_members = dspy.OutputField(desc="The IDs of the agents assigned to the team")
    rationale = dspy.OutputField(desc="The rationale for the team formation")

class Agent(Machine):
    def __init__(self, llm, name, skills=None):
        self.llm = llm
        self.name = name
        self.skills = skills or []
        self.task = None
        states = ['idle', 'assigned', 'executing', 'completed']
        Machine.__init__(self, states=states, initial='idle')
        self.add_transition('assign_task', 'idle', 'assigned')
        self.add_transition('start_execution', 'assigned', 'executing')
        self.add_transition('complete_task', 'executing', 'completed')
        self.add_transition('reset', 'completed', 'idle')

    async def execute_task(self):
        if self.task:
            self.start_execution()
            print(f"{self.name} executing task: {self.task['description']}")
            # Simulate task execution
            prompt = f"{self.name}, imagine you are an AI agent with skills in {', '.join(self.skills)}. Please provide a detailed response on how you would execute the following task: {self.task['description']}. Be specific and provide relevant examples or steps."
            response = self.llm(prompt).pop()
            self.task['execution_result'] = response
            print(f"{self.name} completed task: {self.task['description']} with result: {response}")
            self.complete_task()
            self.task = None

class TeamAllocator:
    def __init__(self, llm, agents: List[Agent]):
        self.llm = llm
        self.agents = agents

    def form_team(self, task_description):
        prompt = f"""
        The task is described as follows: "{task_description}". 
        Form the most suitable team of agents based on their skills. Provide the team member IDs and the rationale for the team formation.

        Available agents and their skills:
        - Agent 1: frontend, backend
        - Agent 2: backend
        - Agent 3: frontend, design

        Provide the response in the following format:
        Team Members: [agent_ids]
        Rationale: [rationale]
        """
        response = self.llm(prompt).pop()
        print(f"Response from Claude: {response}")

        # Parse the response to extract agent_ids and rationale
        lines = response.strip().split("\n")
        agent_ids = []
        rationale = None
        for line in lines:
            if line.strip().startswith("Team Members:"):
                agent_ids = [id.strip() for id in line.strip().split(":")[-1].split(",")]
            elif line.strip().startswith("Rationale:"):
                rationale = line.strip().split(":", 1)[-1].strip()

        if agent_ids and rationale:
            team = [agent for agent in self.agents if agent.name in agent_ids]
            for agent in team:
                if agent.state == 'idle':
                    agent.task = {'description': task_description, 'rationale': rationale}
                    agent.assign_task()
                    print(f"Task '{task_description}' assigned to {agent.name} as part of the team with rationale: {rationale}")
                else:
                    print(f"Agent {agent.name} is already assigned a task. Skipping task allocation.")
        else:
            print("Unable to parse the response and form the team.")

    async def coordinate_execution(self):
        tasks = [agent.execute_task() for agent in self.agents if agent.state == 'assigned']
        await asyncio.gather(*tasks)

# Initialize agents
agents = [
    Agent(llm, name="Agent 1", skills=["frontend", "backend"]),
    Agent(llm, name="Agent 2", skills=["backend"]),
    Agent(llm, name="Agent 3", skills=["frontend", "design"])
]

# Initialize team allocator
team_allocator = TeamAllocator(llm, agents)

# Main execution loop
tasks_to_allocate = [
    "Build the user interface",
    "Set up the database",
    "Design the user experience"
]

async def main():
    for task in tasks_to_allocate:
        team_allocator.form_team(task)
    await team_allocator.coordinate_execution()

asyncio.run(main())
