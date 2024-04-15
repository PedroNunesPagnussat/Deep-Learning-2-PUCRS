# Introdução 


A função de ativação `Softmax` desempenha um papel importante no mundo de Machine Learning (ML),
portanto, é importante garantir que tenhamos uma base sólida sobre essa função antes de mergulharmos em quaisquer variações
como `LogSoftmax` e `Softmax2D` que serão discutidas neste post do blog.


# Softmax


A função `Softmax` transforma um vetor numerico em um distribuição de propabilidadades,
o que nos garante que os elementos do vetor somarão um e que valores maiores terão probabilidades maiores
mapeando cada valor do vetor a uma probabilidade, 
vale ressaltar que está "probabilidade" funciona como uma normalização do vetor e não como a probabilidade em si.
A `Softmax` normaçmente é utilizada na ultima cada de um Multi Layer Perceptron (MLP), 
Em tarefas de classificação multi-classe junto com a [Cross Entropy Loss](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
que nos permite fazer utilizar o truque abaixo,
que nos permite simplificar algumas contas no back propagation

<p align="center"><img src="./imgs/softmax-example.png" alt="Softmax formula" style="height: 512px; width:512px" align="middle"></p>

# Simplificando Cross Entropy

Resumidamente a perda de entropia cruzada é uma função amplamente usada para medir a diferença entre duas distribuições de probabilidade - a distribuição prevista pelo modelo e a distribuição real dos rótulos. Para um caso de classificação multiclasse, é definida como: $$L = - \sum_{c=1}^{M} y_c \log(\hat{y}_c)$$ onde $y_i$ é o vetor de rótulo verdadeiro e $p_i$ é a probabilidade prevista para a classe i calculada pela Softmax.


Agora juntando a cross entropy com a softmax  numa única operação, chamamos isso de Softmax-Cross Entropy Loss. Matematicamente, isso implica plugar a saída da Softmax diretamente na Cross Entropy. No entanto, ao fazer isso, podemos simplificar os cálculos:


<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbg_black%20%5Ccolor%7Bwhite%7D%20L%20%3D%20-%5Csum_%7Bc%3D1%7D%5E%7BM%7D%20y_c%20%5Clog%20%5Cleft%28%5Cfrac%7Be%5E%7Bz_c%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BM%7D%20e%5E%7Bz_j%7D%7D%5Cright%29" alt="LaTeX Equation" align="middle">
</p>


Usando propriedades de logaritmos, podemos reescrever isso como:


$$
L = - \sum_{c=1}^{M} y_c z_c + \log \left(\sum_{j=1}^{M} e^{z_j}\right)
$$

Isso nos da uma vantagem onde podemos simplificar o gradiente para $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$ Esta forma do gradiente significa que a atualização dos pesos durante o backpropagation é proporcional à diferença entre a probabilidade prevista e o verdadeiro rótulo

# LogSoftmax


`LogSoftmax` é uma operação que combina as funções Softmax e logaritmo natural em uma única etapa. O objetivo principal dessa função é converter logits de um modelo em log-probabilidades. Isso é particularmente útil para melhorar a estabilidade numérica e a eficiência computacional em determinadas operações.


A `LogSoftmax` não é nada mais complexo que $\text{LogSoftmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_{j}e^{x_j}}\right)
$ porem podemos utilizar propriedades logaritmas para simplificar essas formula, usando a propriedade do quociente, $\log_b \left( \frac{x}{y} \right) = \log_b(x) - \log_b(y)$ podemos simplificar a formula para $\text{LogSoftmax}(x_i) = x_i - \log\left(\sum_{j}e^{x_j}\right)$.


Vantegens
- *Estabilidade Numérica:* Ao trabalhar diretamente com log-probabilidades, evitam-se problemas de underflow e overflow. 
- *Eficiência Computacional:* Combinar as operações de Softmax e log em uma única função permite otimizar os cálculos, reduzindo a quantidade de trabalho computacional necessário. Isso é mais eficiente do que calcular a Softmax e, em seguida, aplicar o logaritmo aos resultados.
- *Prevenção de Erros Numéricos em Modelos Complexos:* Em modelos de deep learning particularmente complexos ou profundos, pequenos erros numéricos podem se acumular ao longo das camadas. Usar LogSoftmax ajuda a prevenir esses acúmulos, contribuindo para a estabilidade geral do treinamento.
- *Modelagem de Sequências:* No PLN, a LogSoftmax é frequentemente usada em modelos de linguagem e outras tarefas de sequência, onde a eficiência e a estabilidade numérica são críticas devido ao grande tamanho do vocabulário e à complexidade dos modelos.


LogSoftmax é amplamente utilizada em tarefas que envolvem classificação e modelagem de linguagem. Por exemplo, em redes neurais recorrentes (RNNs) e modelos de atenção. Em resumo a LogSoftmax transforma os scores brutos dos modelos em valores que podem ser interpretados de forma probabilística, ao mesmo tempo em que oferece vantagens numéricas e computacionais significativas.

# Softmax2D 

A Softmax2D pode ser entendida como uma extensão da operação de Softmax. Enquanto a Softmax padrão é aplicada a vetores, a Softmax2D é projetada para operar em tensores com duas dimensões significativas. Isso a torna particularmente útil em contextos onde as entradas são imagens ou mapas de características em redes neurais convolucionais


Vantagens
*Trabalho com Estruturas Bidimensionais:* A principal vantagem da Softmax2D é sua habilidade de trabalhar diretamente com dados que são naturalmente bidimensionais, preservando a estrutura espacial dos dados.


Considerações ao utilizar Softmax2D, é crucial considerar o impacto computacional, especialmente em matrizes muito grandes, onde a normalização pode se tornar um gargalo se não for otimizada adequadamente.

Em resumo, a Softmax2D estende os princípios da Softmax para domínios onde as entradas são bidimensionais, oferecendo uma ferramenta a normalização de dados em aplicações de visão computacional, processamento de imagens e análise espacial, mantendo a coesão e significância das estruturas bidimensionais nos dados.

# Mas como isso funciona no Pytorch?

## Softmax

```python
import torch
import torch.nn.functional as F

# Logits simulados para 3 classes, vindo de algum modelo de classificação
logits = torch.tensor([2.0, 1.0, 0.5])

# Aplicando a função softmax para converter logits em probabilidades
probabilities = F.softmax(logits, dim=0)

print("Tensor de probabilidades:", probabilities)

# Verificação: a soma das probabilidades deve ser 1
print("Soma das probabilidades:", probabilities.sum())
```

Isso resultara na seguinte saida:

```output
Tensor de probabilidades: tensor([0.6285, 0.2312, 0.1402])
Soma das probabilidades: tensor(1.0000)
```

## Softmax com CCE

Note que a utilização da softmax não é necessária, pois a implementação da CCE do PyTorch já a aplica por de baixo dos panos.

```python
# Para ver as probabilidades, você pode aplicar explicitamente a softmax aos logits
import torch
import torch.nn as nn
import torch.optim as optim

# Suponha que temos um lote de 4 amostras, cada uma pertencendo a uma de 3 classes possíveis
# Cada amostra é representada por um vetor de características de tamanho 5
inputs = torch.randn(4, 5)
# Rótulos verdadeiros para cada amostra (0, 1, 2 são possíveis classes)
labels = torch.tensor([0, 2, 1, 0])

# Definição de um modelo simples com uma camada linear
model = nn.Linear(5, 3)

# A função Softmax e a Entropia Cruzada são combinadas na seguinte loss function
criterion = nn.CrossEntropyLoss()

# Calcula os logits (saídas antes da softmax)
logits = model(inputs)

# Calcula a perda usando a entropia cruzada, que já inclui a softmax
loss = criterion(logits, labels)

print("Logits:\n", logits)
print("Loss:", loss.item())
```
Isso resultara na seguinte saida:

```output
Logits:
 tensor([[-0.2842, -0.7265, -0.2608],
        [-0.1200,  0.1138,  0.1491],
        [ 1.0189,  0.2717,  0.6223],
        [ 0.0871,  0.7546,  0.8635]], grad_fn=<AddmmBackward0>)
Loss: 1.2823714017868042
```

## LogSoftmax

Note que as soma das log probabilidades não soma 1 como a soma das probabilidades.

```python
import torch
import torch.nn.functional as F

# Logits simulados para 3 classes, vindo de algum modelo de classificação
logits = torch.tensor([2.0, 1.0, 0.5])

# Aplicando a função log softmax para converter logits em log de probabilidades
log_probabilities = F.log_softmax(logits, dim=0)

print("Log Probabilidades:", log_probabilities)

# Verificação: Exponenciando os logaritmos das probabilidades para recuperar as probabilidades originais
probabilities = torch.exp(log_probabilities)
print("Probabilidades from Log Probabilities:", probabilities)

print("Soma das log probabilidades:", sum(log_probabilities))
print("Soma das probabilidades:", sum(probabilities))
```
Isso resultara na seguinte saida:
```output
Log Probabilidades: tensor([-0.4644, -1.4644, -1.9644])
Probabilidades from Log Probabilities: tensor([0.6285, 0.2312, 0.1402])
Soma das log probabilidades: tensor(-3.8931)
Soma das probabilidades: tensor(1.0000)
```

## Softmax 2d

```python
import torch
import torch.nn as nn

# Criando um tensor de exemplo que representa a saída de uma camada convolucional
# Suponha que temos 1 amostra (batch size = 1), 3 canais (classes), altura = 2, largura = 2
logits = torch.tensor([[[[2.0, 1.0], [0.5, 1.2]],
                        [[1.0, 2.0], [0.3, 1.5]],
                        [[0.2, 0.1], [3.0, 1.1]]]])

# Print dos logits para verificação
print("Logits:\n", logits)

# Instanciando Softmax2d
softmax2d = nn.Softmax2d()

print()
print()
# Aplicando Softmax2d nos logits
probabilities = softmax2d(logits)

print("Probabilities:\n", probabilities)

print()
print()
# Verificação: a soma das probabilidades para cada posição espacial deve ser 1
print("Soma das probabilidades:\n", probabilities.sum(dim=1))
```
Isso resultara na seguinte saida:
```output
Logits:
 tensor([[[[2.0000, 1.0000],
          [0.5000, 1.2000]],

         [[1.0000, 2.0000],
          [0.3000, 1.5000]],

         [[0.2000, 0.1000],
          [3.0000, 1.1000]]]])


Probabilities:
 tensor([[[[0.6522, 0.2424],
          [0.0714, 0.3072]],

         [[0.2399, 0.6590],
          [0.0585, 0.4147]],

         [[0.1078, 0.0986],
          [0.8701, 0.2780]]]])


Soma das probabilidades:
 tensor([[[1.0000, 1.0000],
         [1.0000, 1.0000]]])
```

# Conclusão


`Softmax` e suas variações são fundamentais no cenário de deep learning, cada variação servindo a seu propósito único em vários domínios. Enquanto a função softmax fornece uma base ao transformar logits em probabilidades interpretáveis, suas variantes como log-softmax trazem vantagens computacionais e estabilidade numérica, especialmente uteis no cálculo de perda e otimização de gradiente. Por outro lado, softmax2d estende o conceito de softmax para dimensões espaciais, abrindo portas para aplicações avançadas de processamento de imagens, como segmentação.

referencias:

Morgan, P.S. "Softmax." MRI Questions. Available online: https://mriquestions.com/softmax.html.

PyTorch. "torch.nn.LogSoftmax." PyTorch Documentation. Available online: https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html.

Stack Overflow. "How is log-softmax implemented to compute its value and gradient with better numerical stability?" Stack Overflow, May 3, 2020. Available online: https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better.

DeepLizard. "Softmax Function - Clearly Explained with Examples | Deep Learning Basics." YouTube, uploaded by DeepLizard, date not specified. Available online: https://youtu.be/8nm0G-1uJzA?si=tpA4XBsojZciALCB.

S, Abhirami. "Softmax vs LogSoftmax." Medium. Available online: https://medium.com/@AbhiramiVS/softmax-vs-logsoftmax-eb94254445a2.