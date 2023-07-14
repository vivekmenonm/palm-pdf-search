# Project Setup for Accessing Palm API Service Locally

This guide provides the necessary steps to set up your local environment to access the Palm API service.

## Prerequisites

- Google Cloud project with the required permissions and billing enabled.
- Google Cloud SDK installed and authenticated.

## Step 1: Enable the Required APIs

Ensure that the following APIs are enabled in your Google Cloud project:

- Vertex AI API
- Compute Engine API
- Cloud Storage API

You can enable APIs through the Google Cloud Console or by using the following command:


Replace `servicename` with the name of the API you want to enable.

## Step 2: Install and Configure the Cloud SDK

1. Install the Google Cloud SDK by following the instructions in the [Cloud SDK documentation](https://cloud.google.com/sdk/docs/install). Check the turn on screen reader mode checkbox before starting installation

2. Authenticate the SDK by running the following command and selecting the Google account associated with your project:


## Step 3: Set the Default Project
Set the default project: Set your default project for the Cloud SDK by running `gcloud config set project PROJECT_ID`, replacing PROJECT_ID with your required project's ID.

Set your default project for the Cloud SDK by running the following command:
`gcloud auth login`

To authenticate the application default credentials using the gcloud run: `gcloud auth application-default login`


## Usage

After completing the setup steps, you can now access the Palm API service locally using the configured Google Cloud project and SDK.

Remember to run the following command every time you restart your project to authenticate the application default credentials:

`gcloud auth application-default login`