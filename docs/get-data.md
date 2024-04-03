# Data Versioning

* We recommend using [Data Version Control (DVC)](https://dvc.org/) to version your data. 
* DVC is a version control system for data science and machine learning projects. 
* It is designed to handle large files, data sets, machine learning models, and code. 
* DVC works with Git to version control data files and models. 
* If you want to try out the examples using our data, run `dvc pull` in the root directory of the repository.
* Otherwise, see the next section on how to manage your own training data with DVC.


## Downloading the project data

* Make sure your request access to the data by filling out the form [here](https://forms.gle/XEcjEhMDqWU7q9pb6).
* Once you have access, create a service account key by visiting the [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts).
* Create a JSON key for your service account and download it to your local machine. **Note**: Store this key securely.
* Tell the `dvc cli` where to find your downloaded JSON key file as follows:

```bash
dvc remote modify gdrive --local gdrive_service_account_json_file_path <path-to-json-key-file>
```
* Alternatively the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to the path of the JSON key file.
* You are now ready to download the data by running:

```bash
dvc pull

# Note: This will not work if you dont have access to the data
# Request access to the data here: https://forms.gle/XEcjEhMDqWU7q9pb6
```


## Tracking local files with DVC

* You can track your training data with DVC by running `dvc add <path-to-file-or-foler>`.
* For example to track raw data in a file `data.zip`, you can run:
```bash
dvc add data.zip

# Make sure the file is not ignored by git before running the add command
```
* If all is good, your original file will be moved to .dvc/cache and a small metafile `data.zip.dvc` will be created in the original location
* You can now safely commit the `.dvc` file to git. Each time there is a change in the data file, you can run `dvc add data.zip` to update the cache and the `.dvc` file.
* You can either track individual files or entire directories with DVC.

## Saving data to a remote storage

* It is usually a good idea to save your data to a remote storage to avoid losing it in case of a system crash.
* In addition, if you are working in a team, it is easier to share data with your team members.
* DVC supports various remote storage options like AWS S3, Google Cloud Storage, Azure Blob Storage, Google Drive, etc.
* For Example: **To save your data in Google Drive**, you can run:
```bash
dvc remote add -d googledrive gdrive://<google-drive-folder-id>

# You can get the folder id from the URL of the folder in Google Drive
# e.g. https://drive.google.com/drive/folders/1md_0000MvM4AXcB6uzFBHRcZ8oA60000
# The folder id is 1md_0000MvM4AXcB6uzFBHRcZ8oA60000
```
* You can now push your data to Google Drive by running:
```bash
dvc push
```

> **Note:** To use Google Drive as a remote storage, you need to authenticate DVC with Google Drive. The recommended 
> way is to create a [Service Account](https://cloud.google.com/iam/docs/service-account-overview) from [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts). 
> Once your service account is created, you can create and download the [JSON key file](https://cloud.google.com/iam/docs/keys-create-delete#console) and set the `GOOGLE_APPLICATION_CREDENTIALS` to the path of the JSON key file.


## Case Study: Versioning data for Data Science Teams

* Although at first, it seems like a good idea to store data in a shared folder on Google Drive, you will soon realize that it is not a scalable solution.
* You can potentially get away with it for small projects where the data can be downloaded quickly.
* However, once you start working with GigaBytes or TeraBytes of data, downloading the data every time you need to train will become a huge bottleneck.
* For such case, tools like DVC come to the rescue.
* Although DVC supports Google Drive as a remote storage, it is not recommended for large datasets. Instead, you can use cloud storage services like AWS S3, Google Cloud Storage, Azure Blob Storage, etc.
* Assume the team has decided to use Azure Blob Storage to store the data to develop a custom [BERT](https://arxiv.org/abs/1810.04805) model for classifying court cases.
* The team has created a storage account on Azure and has the following details:
  - Storage Account Name: `mydatastore`
  - Container Name: `court-cases`
  - Access Key: `my-access-key`
- The team can now add the Azure Blob Storage as a remote storage in DVC by running:
```bash
dvc remote add -d myazure azure://mydatastore/court-cases

# This assumes that the access key is stored in the environment variable AZURE_STORAGE_KEY
# Refer to the DVC documentation for other ways to authenticate with Azure Blob Storage
```
* Now the team can push all the tracked files and folders to Azure Blob Storage by running:
```bash
dvc push -r myazure

# Note you could have multiple remotes and you can specify the remote name with -r
``` 
* Lets say a new team member joins the team and wants to work on the project. The team member can pull the data from Azure Blob Storage by running:
```bash
export AZURE_STORAGE_KEY=my-access-key
dvc pull -r myazure

# This pulls all the data from the remote storage to the local machine
# However, in practice, you may want to pull only the data you need for your task to save time and space
# You can do this by specifying the path to the data you want to pull. For example:

# dvc pull -r myazure data/raw/court-cases-proceedings.csv
```