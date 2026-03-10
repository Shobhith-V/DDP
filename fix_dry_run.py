with open('/home/shobs/Desktop/DDP/exp3_retry_revised_org__shobhith_code_10.py', 'r') as f:
    text = f.read()

# Dry run test just modifying config to run real fast to check execution graph stability
text = text.replace('"heart_epochs": 25000', '"heart_epochs": 25')
text = text.replace('"brain_epochs": 100', '"brain_epochs": 2')
text = text.replace('"mlp_epochs": 200', '"mlp_epochs": 5')

with open('/home/shobs/Desktop/DDP/dry_run_test.py', 'w') as f:
    f.write(text)
