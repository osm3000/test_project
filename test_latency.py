import requests
import timeit


payload = {
    'data': "Hello from the other side of the world",
    'quantized_model': True
}

# print(timeit.timeit(requests.get("http://127.0.0.1:5000/get_classification?data=I%20am%20weak")), 10)

# code snippet to be executed only once
mysetup = "import requests"

# code snippet whose execution time is to be measured


mycode = '''
requests.get("http://127.0.0.1:5000/get_classification?data=I%20am%20weak&quantized_model=false")
'''

# timeit statement
print(timeit.timeit(setup=mysetup,
                    stmt=mycode,
                    number=1000))


mycode = '''
requests.get("http://127.0.0.1:5000/get_classification?data=I%20am%20weak&quantized_model=true")
'''

# timeit statement
print(timeit.timeit(setup=mysetup,
                    stmt=mycode,
                    number=1000))
