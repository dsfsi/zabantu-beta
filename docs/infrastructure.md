# Infrastructure

* We have provission a set of getting started [Infrastucture As Code(IAC)](https://en.wikipedia.org/wiki/Infrastructure_as_code) scripts to help you get started with the infrastructure setup.
* You can find the scripts inside the `infra` directory.
* Currently we offer:
  * [AWS EC2 Instance](https://aws.amazon.com/ec2/) under `infra/aws_dvsm.tf`
  * [Google Cloud Compute Engine](https://cloud.google.com/compute/) under `infra/gcp_dvsm.tf`
  * [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) under `infra/azure_dvsm.tf`. This machine is pre-configured with the necessary NVIDIA GPU drivers and packages required for most deep learning frameworks.

   
## Deployment to AWS with Terraform

### Pre-requisites

Before deploying the infrastructure to AWS using Terraform, ensure you have the following pre-requisites:

1. [Terraform](https://www.terraform.io/downloads.html) installed on your local machine.
2. An [AWS account](https://aws.amazon.com/console) with the necessary permissions to create and manage EC2 instances.
3. [AWS CLI](https://aws.amazon.com/cli/) installed and configured with the necessary credentials.

### Steps

1. Clone the repository to your local machine:
```bash
# NB: SKIP THIS STEP if you have already cloned the repository

git clone https://github.com/ndamulelonemakh/zabantu-beta.git
```

2. Navigate to the `infra` directory:
```bash
cd zabantu-beta/infra
```
3. Configure your AWS credentials using the AWS CLI:

```bash
aws configure
```

> **Note:** You will need to create an IAM user with programmatic access and attach the necessary permissions to create and manage EC2 instances. The access key and secret key generated for the IAM user will be used to configure the AWS CLI.

4. Review and modify the `aws_dvsm.tf` file according to your requirements. Update the instance type, region, and any other relevant configurations.
5. Initialize the Terraform working directory:
```bash
terraform init
```

6. Preview the changes that Terraform will make:
```bash
terraform plan
```

7. If the plan looks good, apply the changes to provision the EC2 instance:
```
terraform apply
```

   Confirm the changes by typing "yes" when prompted.

8. Once the provisioning is complete, Terraform will output the public IP address of the EC2 instance. Use this IP address to connect to the instance via SSH.

```bash
ssh ubuntu@<public-ip-from-terraform-output>
```

<br/>

## Deployment to Google Cloud with Terraform

### Pre-requisites

Before deploying the infrastructure to Google Cloud using Terraform, ensure you have the following pre-requisites:

1. [Terraform](https://www.terraform.io/downloads.html) installed on your local machine.
2. A Google Cloud account with the necessary permissions to create and manage Compute Engine instances.
3. Google Cloud Service Account and a JSON Key File for authentication.

### Steps

1. Clone the repository to your local machine:
```bash
# NB: SKIP THIS STEP if you have already cloned the repository

git clone https://github.com/ndamulelonemakh/zabantu-beta.git
```

2. Navigate to the `infra` directory:
```bash
cd zabantu-beta/infra
```

3. Configure your Google Cloud credentials by setting the path to the service account key file in the `GOOGLE_APPLICATION_CREDENTIALS` environment variable, or by using the `gcloud` command-line tool to authenticate.
```bash
# Example for authenticatiing using gcould

gcloud auth activate-service-account --key-file=service-account-key.json
```

4. Review and modify the `gcp_dvsm.tf` file according to your requirements. Update the instance type, region, and any other relevant configurations.
5. Initialize the Terraform working directory:
```bash
terraform init
```

6. Preview the changes that Terraform will make:
```bash
terraform plan
```

7. If the plan looks good, apply the changes to provision the Compute Engine instance:
```bash
terraform apply
```

   Confirm the changes by typing "yes" when prompted.

8. Once the provisioning is complete, Terraform will output the public IP address of the Compute Engine instance. Use this IP address to connect to the instance via SSH.


<br/>


## Deployment to Azure with Bicep

[Bicep](https://github.com/Azure/bicep) is a Domain Specific Language (DSL) for deploying Azure resources declaratively. 
It aims to drastically simplify the authoring experience with a cleaner syntax and better support for modularity and code re-use compared to ARM templates.

### Pre-requisites

Before you can use the provided Bicep scripts to deploy your infrastructure on Azure, you need to have the following pre-requisites:

1. **Azure Account**: You need to have an Azure account. If you don't have one, you can create it from the [Azure portal](https://portal.azure.com/).

2. **Azure CLI**: You need to have the Azure CLI installed on your local machine. You can download it from the [official website](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli). Make sure to install a version that is compatible with the scripts provided in this repository.

3. **Bicep CLI**: You need to have the Bicep CLI installed on your local machine. You can download it from the [official GitHub repository](https://github.com/Azure/bicep/blob/main/docs/installing.md).

### Authentication

To authenticate with Azure, you can use the Azure CLI. Run the following command and follow the instructions:

```bash
az login

# If you are using a managed identity, you can run the following command
#az login --identity

# Or if you want to opt for device login
#az login --use-device-code
```

- This will open a new window in your default web browser where you can log in with your Azure credentials. 
Once you're logged in, the CLI will be able to manage resources in your Azure account.

### Deployment

- To deploy your infrastructure with Bicep, run the following command:

```bash
az group create --name myrg100 --location westeurope
az deployment group create --name MyGPUVmDeplyment0010 \
                           --resource-group myrg100 \
                           --template-file infra/azure_dvsm.bicep \
                           --parameters location=westeurope adminUsername=ubuntu vmName=myT4Vm010
```

- Replace `myrg100`, `MyGPUVmDeplyment0010`, `westeurope`, `ubuntu`, and `myT4Vm010` with your desired settings.


# Single Click Deployment

- You can also deploy the infrastructure directly from the GitHub repository using the "Deploy to Azure" button.
- Click on the button below to deploy the infrastructure to Azure:

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fndamulelonemakh%2Fzabantu-beta%2Fmain%2Finfra%2Fazure_dvsm.bicep)

- Similarly, for AWS:

[![Deploy to AWS](https://

- And GCP:

[![Deploy to GCP](https://
