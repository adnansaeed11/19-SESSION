import dagshub
import mlflow

mlflow.set_tracking_uri('https://dagshub.com/adnansaeed11/19-SESSION.mlflow')

dagshub.init(repo_owner='adnansaeed11', repo_name='19-SESSION', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)