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

Here’s an optimized version with that detail:

- **Discussion Date:** Date of the exchange between the MP and Ministry.
- **Ministry:** The Ministry responsible for the debate, represented by its Minister, who serves as the responder (e.g., Science, Technology and Innovation; Sport, Arts, and Culture, etc.). The Ministry also guides the focus of the discussion.
- **Member of Parliament:** Primarily opposition MPs, who act as questioners, posing probing questions to the Minister. MPs from the ruling party may also participate in responding to these questions.

The dataset was sourced from [PMG’s website](https://pmg.org.za/) using Selenium and Python, as detailed in the `nbs` folder. A smaller sample of this data is utilized for development purposes. This dataset forms the foundation for the PMG Assistant’s recommendations and guidance capabilities.

You can find the data in [`data/pmg_data.parquet.brotli`](data/pmg_data.parquet.brotli).

## Tech stack

- Python
- Docker
- [Minsearch](https://github.com/alexeygrigorev/minsearch) for full-text search
- Flask as the API interface
- Grafana for monitoring and PostgreSQL
- OpenAI as an LLM


## Setting Up OpenAI API Key and Dependency Management

To use OpenAI, an API key is required. Begin by installing `direnv` (on Ubuntu, run `sudo apt install direnv` followed by `direnv hook bash >> ~/.bashrc`). Then, copy `.envrc_template` to `.envrc` and add your API key. For security, it’s recommended to create a new OpenAI project and use a dedicated key. Run `direnv allow` to load the key into your environment.

For dependency management, we use `pipenv`. Install it by running `pip install pipenv`, then set up the app dependencies with `pipenv install --dev`.

## Running the application

**Database Configuration**

Before launching the application for the first time, it is essential to initialize the database. Begin by starting the PostgreSQL service with the command `docker-compose up`. Once the database is running, enter the project environment using `pipenv shell`, navigate to the `pmg-assist` directory, and set the `POSTGRES_HOST` environment variable to `localhost` by running `export POSTGRES_HOST=localhost`. Finally, execute the database preparation script with `python pmg_db_prep.py` to complete the setup.

**Running with Docker (Without Compose)**

In some cases, you may prefer to run the application in Docker without using Docker Compose, such as for debugging purposes. Start by preparing the environment with Docker Compose as outlined in the previous section. Once that is set up, build the Docker image using the command `docker build -t pmg-assist .`. After the image has been built, run the application with the following command: 

```bash
docker run -it --rm \
    --network="pmg-assist_default" \
    --env-file=".env" \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    -e DATA_PATH="data/pmg_data.parquet.brotli" \
    -p 5000:5000 \
    pmg-assist
```

This command will execute the application within the specified network while passing in the necessary environment variables.

**Using the Application**

Once the application is up and running, you can start interacting with it. For command-line interface (CLI) interactions, we've developed an interactive CLI application using the Questionary library. To initiate the CLI, execute the command `pipenv run python pmg_cli.py`. If you want the application to randomly select a question from our ground truth dataset, use the command `pipenv run python pmg_cli.py --random`.

Additionally, you can utilize the `requests` (W.I.P)

For those who prefer to interact with the API via curl (W.I.P)

Upon execution, you will receive a response similar to the following:

```json
{
    "answer": ". . .",
    "conversation_id": ". ..",
    "question": ". . .?"
}
```