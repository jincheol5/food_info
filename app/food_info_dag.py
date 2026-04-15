from airflow import DAG
from modules import DBInterface

dag=DAG(
    dag_id="nutrition_extractor",
    schedule=None
)

