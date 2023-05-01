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

"text" : "Developed by Game Freak and published by Nintendo, the Pokémon series began in Japan in 1996, and features several species of creatures called \"Pokémon\" that players, called \"trainers\", are encouraged to capture, train, and use to battle other players' Pokémon or interact with the game's world. Pikachu was one of several different Pokémon designs conceived by Game Freak's character development team. Artist Atsuko Nishida is credited as the main person behind Pikachu's design, which was later finalized by artist Ken Sugimori. According to series producer Satoshi Tajiri, the name is derived from a combination of two Japanese onomatopoeia: ピカピカ (pikapika), a sparkling sound, and チューチュー (chūchū), a sound a mouse makes. Despite its name's origins, however, Nishida based Pikachu's original design, especially its cheeks, on squirrels. Developer Junichi Masuda noted Pikachu's name as one of the most difficult to create, due to an effort to make it appealing to both Japanese and American audiences. Pikachu was designed around the concept of electricity. They are creatures that have short, yellow fur with brown markings covering their backs and parts of their lightning bolt-shaped tails. They have black-tipped, pointed ears and red circular pouches on their cheeks, which can spark with electricity. They attack primarily by projecting electricity from their bodies at their targets. Within the context of the franchise, Pikachu can transform, or \"evolve,\" into a Raichu when exposed to a \"Thunder Stone.\" Pikachu was originally planned to have a second evolution called Gorochu, which was intended to be the evolved form of Raichu. In Pokémon Gold and Silver, \"Pichu\" was introduced as an evolutionary predecessor to Pikachu. In Pokémon Diamond and Pearl, gender differences were introduced; since those games, female Pikachu have an indent at the end of their tails, giving the tail a heart-shaped appearance. Initially, both Pikachu and fellow Pokémon Clefairy were chosen to be lead characters for the franchise merchandising, with the latter as the primary mascot to make the early comic book series more \"engaging\". Production company OLM, Inc. suggested Pikachu as the mascot of the animated series after finding that Pikachu was popular amongst schoolchildren and could appeal to both boys and girls, as well as their mothers. Pikachu resembled a familiar, intimate pet, and yellow is a primary color and easier for children to recognize from a distance. Additionally, the only other competing yellow mascot at the time was Winnie-the-Pooh. Pikachu's design has evolved from its once-pudgy body to having a slimmer waist, straighter spine, and more defined face and neck; Sugimori has stated these design changes originated in the anime, making Pikachu easier to animate, and were adopted to the games for consistency. \"Fat Pikachu\" was revisited in Pokémon Sword and Shield, where Pikachu received a Gigantamax Form resembling its original design."

}

