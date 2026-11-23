"""A stand-in for the orchestrator, its knowledge base and the LLM, to test the sandbox in the use cases.

None of this is part of the sandbox: in the system it is the orchestrator's job. It is kept as small as possible,
to show what the sandbox makes possible:

- knowledge.py    assets (which microgrid offers which capability, as data or as a model) and recipes (how a
                  capability is derived from other capabilities with operators)
- templates.py    filling a recipe's chain with the assets of one microgrid
- generator.py    the LLM: writes a recipe when a capability is missing (here: a canned reply per capability,
                  checked against the operator catalogue of the sandbox)
- sandbox.py      the sandbox over HTTP
- orchestrator.py UC1: one step of a workflow (a capability over a scope of microgrids); a missing capability is
                  derived, or generated, checked, registered and reused, instead of breaking the workflow
- adaptation.py   UC3: the best demand model for a microgrid with little history, among training it locally,
                  reusing the model of a similar microgrid, or adapting that model
"""
