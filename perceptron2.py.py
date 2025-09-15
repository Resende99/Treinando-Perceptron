# import -> incorpora uma biblioteca \
# (código já escrito para resolver algum problema específico)
import numpy as np

class Perceptron:
    # Declaração do construtor da classe
    def __init__(self):
        pass

    def activation(self, x):
        # Função de ativação sigmoid
        return 1 if x >= 0 else 0

    def train(self, inputs, outputs, learning_rate=0.5, epochs=5):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Inicialização de três pesos e bias
        w1, w2, w3, bias = np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)

        for i in range(epochs):
            for j in range(len(inputs)):
                # Usando o método de ativação
                z = w1 * inputs[j][0] + w2 * inputs[j][1] + w3 * inputs[j][2] + bias
                sigmoid = self.activation(z)

                # atualização dos pesos por iteração
                w1 = w1 + learning_rate * (outputs[j][0] - sigmoid) * inputs[j][0]
                w2 = w2 + learning_rate * (outputs[j][0] - sigmoid) * inputs[j][1]
                w3 = w3 + learning_rate * (outputs[j][0] - sigmoid) * inputs[j][2]
                bias = bias + (learning_rate * (outputs[j][0] - sigmoid))

        return w1, w2, w3, bias

    def predict(self, weights, x1, x2, x3):
        # Usando o método de ativação
        z = x1 * weights[0] + x2 * weights[1] + x3 * weights[2] + weights[3]
        return 1 if self.activation(z) > 0.5 else 0


if __name__ == '__main__':
    # Entradas das portas lógicas (em pares)
    inputs = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
    outputs = [[0], [0], [0], [1]]

    print(' ')
    for i in range(len(inputs)):
        print(inputs[i][0], inputs[i][1], inputs[i][2], '->', outputs[i][0])


    print(' ')
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            print(inputs[i][j], end=', ')
        print(' ')


    perceptron = Perceptron()

    # Treinando o perceptron
    tanning = perceptron.train(inputs=inputs, outputs=outputs, learning_rate=0.1, epochs=100)

    # Testando para todas as entradas
    print('\nResultados para todas as entradas:')
    for entrada in inputs:
        resultado = perceptron.predict(tanning, entrada[0], entrada[1], entrada[2])
        print(f'{entrada[0]}, {entrada[1]}, {entrada[2]} -> {resultado}')

