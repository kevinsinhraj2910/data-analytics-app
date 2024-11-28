# data-analytics-app
## Project Overview
This project is a Python-based data analytics application designed to analyze data and provide insights. It utilizes Docker for containerization, Kubernetes for orchestration (via Minikube), and Jenkins for a CI/CD pipeline. The application is thoroughly tested using Pytest and Selenium (optional for web UI).

## Setup Instructions
Clone the repository to your local machine.
Install the required dependencies listed in the requirements.txt file.
Build and containerize the application using Docker.
Deploy the application to a Kubernetes cluster using Minikube.
## Testing
The project includes unit tests written with Pytest. Selenium-based tests are optional for validating the web interface if applicable.

## Deployment
Minikube is used to orchestrate the application. Kubernetes manifests define the deployment and service configurations for the application. Access the application through the exposed service endpoint.
