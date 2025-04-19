## Aim
- Test CAG

## Process
- Generate made up stuff about a topic
    - using gemma qna, for each para, get it in a json training data format
- To take a small model
- see what it knows about the made up stuff
    - qna
- Use the cag to put this made up stuff in the small model
- test if its better at answering stuff about this made up stuff
- perfromance : speed, accuracy
- If possible do a rag thing and comapre the same

## Run
 .\venv\Scripts\activate 
python .\code\kvcache.py --kvcache file --similarity bertscore --output "./output/result_1.txt"
python .\code\kvcache.py --kvcache file --similarity bertscore --output "./output/result_llama.txt" --model "B:\\Work\\Code\\f\\4\\testing\\llm\\models\\unsloth-Llama-3.2-1B-Instruct"

## TODO
1. speed : run questions via non chaced and compare with cached one
    - check why the answers are useless
2. accuracy : make a rag and compare with cag

## Process
- kvcache.py is long and complicated
- use /etc/cagdemo_llama_cpu.ipynb