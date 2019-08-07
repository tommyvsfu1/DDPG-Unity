# Continuous Control 
This is platform from Unity https://arxiv.org/pdf/1809.02627.pdf, just like openai gym.  
It's a 2-joint robot arm problem in continuous action space.


## DDPG

Solved!
![](https://i.imgur.com/ejDkuGS.png)
![](https://i.imgur.com/Vx9d9Pw.png)
### Eval
on Mac
```python
python eval.py 
```
on gcp
```python
python eval.py --machine=s
```

## Train
on Mac
```python
python run.py 
```
on gcp
```python
python run.py --machine=s
```