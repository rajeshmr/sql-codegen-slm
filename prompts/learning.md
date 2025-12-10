## Learning Questions 
- Why do we put the schema in the user message rather than in the system message? 
- What's the advantage of this approach for dynamic schema handling?
- Why do we need a separate test set if we already have a validation set?
- What would happen if we didn't stratify the split by complexity?
- Why create demo schemas manually instead of using examples from Spider?
- What's the difference between a checkpoint and the final model? Why do we save checkpoints every 500 steps?
- What was the final training loss after 20 steps? Why is it still relatively high (probably 0.8-1.5) even though the model is "learning"?