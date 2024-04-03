# Pre-training Data

## Raw Data

* For pre-training our LLMs, we use a curated data source for 9 Bantu languages namely: `isiZulu`, `isiXhosa`, `Sepedi`, `Setswana`, `Sesotho`, `Xitsonga`, `siSwati`, `Tshivenda` and `Shona`.
* Each language is represented by a separate text file.

## Processed Data

* We use the raw version as well as the processed version of the data for training our models.
* The processed version contains data that is `normalized` and `lowercased`

## File Structure

* The data is stored in the `data` directory in the root of the repository.
* The folder structure is as follows:
    ```
    data
    ├── interim/train
    │   ├── zul.txt
    │   ├── xho.txt
    │   ├── nso.txt
    │   ├── tsn.txt
    │   ├── sot.txt
    │   ├── tso.txt
    │   ├── swa.txt
    │   ├── ven.txt
    │   └── sna.txt
    └── processed/train  # uncased
        ├── zul.txt
        ├── xho.txt
        ├── nso.txt
        ├── tsn.txt
        ├── sot.txt
        ├── tso.txt
        ├── swa.txt
        ├── ven.txt
        └── sna.txt
    ```
## Data Versioning

* We use [Data Version Control (DVC)](https://dvc.org/) to manage our data.

## Data Access

* Due to the large size of the data, we use Azure Blob Storage as our primary remote storage.
* A secondary Google Drive remote storage is used for sharing data with collaborators.
* Note that due to [Copyright](https://www.microsoft.com/en-us/legal/intellectualproperty/copyright) restrictions, we are unable to share the data publicly.
* However, you can send a request to access the data storedin Google Drive by contacting us at [info@mungana.com](mailto:info@mungana.com).
* Note that you will need a Google account to access the data.
* Once the data is shared with you, you can create a [Service Account](https://cloud.google.com/iam/docs/service-account-overview) from [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts) and set the `GOOGLE_APPLICATION_CREDENTIALS` to the path of the JSON key file.
* You can then add the Google Drive remote storage to DVC by running:
```bash
dvc remote add -d googledrive gdrive://<google-drive-folder-id>
```
* Finally, you can pull the data by running:
```bash
dvc pull -r googledrive
``` 
* You can also use the Azure Blob Storage as a remote storage by running:
```bash
dvc remote add -d myazure azure://mungana-public-assets/zabantu-datastore
dvc pull -r myazure

# Note: You will also need to request access to get this data
# You will need an Azure account to access the data through RBAC
```

---

# Fine-tuning Data

* You can fine-tune the model on any downstream task using a dataset of your choice. This includes:
    - Sentiment Analysis
    - Named Entity Recognition
    - Text Classification
    - Question Answering
    - Language Modeling
    - And many more...
  
* For demo purposes we evaluate the trained models using published datasets from the [Hugging Face Datasets](https://huggingface.co/datasets) library.
    - [PuoBERTA News](https://huggingface.co/datasets/puoberta-news)
    - [ZaBantu News](https://huggingface.co/datasets/zabantu-news)
    - [Masakhane NER](https://huggingface.co/datasets/masakhane-ner)
  
* You can also fine-tune the model on your own dataset provided it is in the correct format. i.e. CSV, JSON or Parquet
* Due to the scarcity of high-quality labeled data for Bantu languages, we recommend thta you contribute to the [Masakhane](https://www.masakhane.io/) project by providing labeled data for your language.










