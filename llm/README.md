# PMG Assistant

Staying informed with parliament routines is challenging,
especially for the comon person. Watching TV can be intimidating, and personal
political analysts aren't always available.

The PMG Assistant provides a conversational AI that helps
users understand parlimentary discussions and find alternatives, making PMG understanding more
manageable.


## Project overview

The PMG Assistant is a RAG application designed to assist
users with PMG undertanding.

The main use cases include:

1. Ministry Selection: Recommending topics based on the type of activity, targeted sectors in discussion, or available Ministers and PMs.
2. Ministry Replacement: Replacing an Minister/MP with suitable alternatives.
3. Conversational Interaction: Making it easy to get information without sifting through youtube, TV programmes, manuals or websites.

## Dataset

The dataset used in this project contains information about
various interactions betweeb MPs and Ministers, including:

- **Discussion date:** The name of the exercise (e.g., Push-Ups, Squats).
- **Ministry:** The general category of the exercise (e.g., Strength, Mobility, Cardio).
- **Member of Parliament:** The equipment needed for the exercise (e.g., Bodyweight, Dumbbells, Kettlebell).
- **Minister:** The part of the body primarily targeted by the exercise (e.g., Upper Body, Core, Lower Body).


The dataset was generated using ChatGPT and contains 207 records. It serves as the foundation for the Fitness Assistant's exercise recommendations and instructional support.

You can find the data in [`data/data.csv`](data/data.csv).


## Tech stack

- Python
- Docker
- [Minsearch](https://github.com/alexeygrigorev/minsearch) for full-text search
- Flask as the API interface
- Grafana for monitoring and PostgreSQL
- OpenAI as an LLM