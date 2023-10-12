# Skills Tagger

# Requires Docker
# Requires OpenAI Key in a config.py, named "api_key"
# Python v3.8+

# Step 1: Configure Weviate

- download YAML file using following curl command: 
- curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.18.0"

Open docker-compose.yml and add the following volume:

image: semitechnologies/weaviate:1.18.0
volumes:
  - /var/weaviate:/var/lib/weaviate

Save the yml file

Run docker-compose up-d

# Step 2: Python package install

Install the following packages: 

pip install openai
pip install pandas
pip install weaviate-client

