# Terraform — AWS S3 remote pour DVC

Ce module crée un bucket S3 privé pour stocker les données et artefacts suivis par DVC.

## Ce que le module fait

- crée un bucket S3 dédié ;
- bloque l'accès public ;
- active le chiffrement serveur AES256 ;
- peut activer le versioning ;
- ajoute des tags simples.

## Utilisation

```bash
cd terraform/aws_dvc_remote
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

## Variables principales

- `aws_region`
- `bucket_name`
- `enable_versioning`
- `environment`
- `project`

## Une fois le bucket créé

Depuis la racine du projet :

```bash
dvc remote add -d s3remote s3://<bucket-name>
dvc remote modify s3remote region <aws-region>
dvc push
```

## Exemple

```bash
dvc remote add -d s3remote s3://medvision-dvc-prod
dvc remote modify s3remote region eu-west-3
dvc push
```
