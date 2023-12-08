import requests


def test_prompt_endpoint():
    url = 'http://localhost:4002'
    # Test the prompt endpoint
    target_output = 'simple, lively, strong'
    response = requests.post(url + '/prompt', json={'target_output': target_output})
    response_json = response.json()
    print(response_json)

    # Use pytest's assertion style
    assert 'prompt' in response_json, "Response does not contain 'prompt'"
    assert 'score' in response_json, "Response does not contain 'score'"
    assert isinstance(response_json['prompt'], str), "'prompt' is not a string"
    assert isinstance(response_json['score'], (int, float)), "'score' is not a number"
    assert len(response_json['prompt']) > 0, "'prompt' is empty"

def main():
    #run test:
    test_prompt_endpoint()
    
if __name__=='__main__':
    main()
