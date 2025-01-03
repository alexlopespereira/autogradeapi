# Faca aqui a validacao do Exercicio 2.4




# Fazer uma requisição HTTP
r = requests.get("https://www.servicos.gov.br/api/v1/servicos/orgao/315")
# Transformar a resposta em JSON e pegar a lista de serviços cuja chave é "resposta"
servicos = r.json()["resposta"]
# Imprimir o primeiro serviço da lista
print(servicos[0])

"""### 2.11 Percorra a lista ```servicos``` selecionando os atributos de interesse ```nome, sigla, servicoDigital, nomeOrgao```. Construa o atributo ```id_servico``` a partir do atributo ```id``` coletando o número (id do serviço) que aparece depois da ultima barra da URL do serviço. Construa um dicionário de dicionários. No dicionário externo as chaves são o ```id_servico```, e seu respectivo valor deve ser um dicionário com os atributos de interesse.

#### Dica 1: Consulte o conteúdo da requisição que retorna o dicionário de serviços (veja as curiosidades apresentadas acima). E a partir da inspeção do conteúdo escolha o nome das chaves do dicionário cujo valor deseja acessar. Essas chaves coincidem com o nome dos atributos do enunciado.
#### Dica 2: O nome do órgão não está no primeiro nível do dicionário. Para acessar o nome, acesse um dicionário por meio da chave ```"orgao"``` e dentro dele haverá uma chave ```"nomeOrgao"```. Ou seja ```dicionario["orgao"]["nomeOrgao"]```. Confira isso por meio de uma inspeção visual.
"""

# Faca aqui seus testes do Exercício 2.11

# Faca aqui sua função do Exercício 2.11
def atributos_interesse(serv):
    # YOUR CODE HERE
    raise NotImplementedError()

# Faca aqui a validacao do Exercicio 2.11
atr_inter = atributos_interesse(servicos)
entradas = [[servicos]]
saidas = [{"nome": "Solicitar diagnóstico de referência em chikungunya",
  "sigla": "LABFLA", "servicoDigital": False, "nomeOrgao": "Fundação Oswaldo Cruz (FIOCRUZ)"}]
validate(atributos_interesse, entradas, lambda x: x["9031"], saidas, "2.11")

"""### 2.12 Crie uma função que receba como argumento o dicionário do exercício anterior e calcule a porcentagem de serviços digitais presentes no dicionário."""

# Faca aqui seus testes do Exercício 2.12

# Faca aqui sua função do Exercício 2.12
def calc_porcentagem_digital(servs):
    # YOUR CODE HERE
    raise NotImplementedError()

# Faca aqui a validacao do Exercicio 2.12
atr_inter = atributos_interesse(servicos)
entradas = [[atr_inter]]
saidas = [0.24]
validate(calc_porcentagem_digital, entradas, lambda x: round(x, 2), saidas, "2.12")

