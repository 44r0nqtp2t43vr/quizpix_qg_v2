Docker base image and installation details: https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker


1. Build create image and run docker locally:

docker build -t question_generation .
docker run --name quizpix -p 80:80 question_generation
docker run --name quizpix -p 80:80 --cpus="4" --memory=32768m question_generation

2. Postman Instructions:

http://127.0.0.1/generatequiz

{

"text" : "In 1892 Filipinos interested in the overthrow of Spanish rule founded an organization following Masonic rites and principles to organize armed resistance and terrorist assassinations within a context of total secrecy. It operated as an alternative Filipino government complete with a president and cabinet. When Andrés Bonifacio assumed control over the organization, it became much more aggressive. With the Grito de Balintawak, the Philippine revolution began. Filipinos ripped up their tax and citizenship documents and started fighting through Luzon. Emilio Jacinto commanded Katipunan's troops in several decisive struggle where both sides sustained major losses. The Katipunan movement frightened the Spanish and their supporters in the country. Consequently, the authorities arrested or exiled some 4,000 rebels, not to mention the myriad executions. At this time, the Filipinos were by no means united; Emilio Aguinaldo served as president of the insurgent government while José Rizal headed the Liga Filipina. When General Camilo de Polavieja became the new Spanish military governor on December 3, 1896, he utilized the same strategy of reconcentration as did his counterpart Valeriano Weyler in Cuba. He also ordered the execution of Rizal and 24 others. The spanish crackdown led to a series of victories against Andrés Bonifacio and the Katipunan that Aguinaldo was quick to take advantage of at the Tejeros Convention in March 1897 to force the Katipunan into his new revolutionary government. The Katipunan was revived briefly during the insurrection against the U.S. in 1900."

}


3. Google cloud Run: Push docker to Google container registry

Create a project on gcloud console.
Install Gcloud SDK from https://cloud.google.com/sdk/docs/quickstart
gcloud init

docker build . --tag gcr.io/quizpix-379716/quizpix:latest


https://cloud.google.com/container-registry/docs/advanced-authentication
gcloud auth configure-docker

docker push gcr.io/quizpix-379716/quizpix:latest

4. Deploy API using Google Cloud Run

gcloud init  ---> Choose re-initialize this configuration [default] with new settings --> Pick correct cloud project to use.


Parameters: https://cloud.google.com/sdk/gcloud/reference/run/deploy

gcloud run deploy --image gcr.io/quizpix-379716/quizpix:latest --cpu 2 --concurrency 1 --memory 8Gi --platform managed --min-instances 1 --timeout 10m --port 80


5. Postman Instructions:

https://quizpix-bno7lgu4mq-as.a.run.app

{

"text" : "In 1892 Filipinos interested in the overthrow of Spanish rule founded an organization following Masonic rites and principles to organize armed resistance and terrorist assassinations within a context of total secrecy. It operated as an alternative Filipino government complete with a president and cabinet. When Andrés Bonifacio assumed control over the organization, it became much more aggressive. With the Grito de Balintawak, the Philippine revolution began. Filipinos ripped up their tax and citizenship documents and started fighting through Luzon. Emilio Jacinto commanded Katipunan's troops in several decisive struggle where both sides sustained major losses. The Katipunan movement frightened the Spanish and their supporters in the country. Consequently, the authorities arrested or exiled some 4,000 rebels, not to mention the myriad executions. At this time, the Filipinos were by no means united; Emilio Aguinaldo served as president of the insurgent government while José Rizal headed the Liga Filipina. When General Camilo de Polavieja became the new Spanish military governor on December 3, 1896, he utilized the same strategy of reconcentration as did his counterpart Valeriano Weyler in Cuba. He also ordered the execution of Rizal and 24 others. The spanish crackdown led to a series of victories against Andrés Bonifacio and the Katipunan that Aguinaldo was quick to take advantage of at the Tejeros Convention in March 1897 to force the Katipunan into his new revolutionary government. The Katipunan was revived briefly during the insurrection against the U.S. in 1900."

}

