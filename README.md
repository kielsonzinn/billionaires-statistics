## [Link Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset)

## Participantes

- Cristina Luiza Pelepenko, 2034930
- João Pedro Albano Menegati, 2033518
- Kielson Zinn da Silva, 2003317


## [Builder](https://refactoring.guru/pt-br/design-patterns/builder)

Criado builder para facilitar a criacao de display form, ou seja ira usar um builder para montar a tela, passando os dados do dataset, em seguida os atributos necessarios, como titulo, descricao entre outros atributos.

A alteração pode ser vista no seguinte [commit](https://github.com/kielsonzinn/billionaires-statistics/commit/805bdd62fa953bbfa18418c4fe35f83a652d5bd8)

## [Template method](https://refactoring.guru/pt-br/design-patterns/template-method)

Criado template method para apresentar a tela, aonde cada classe deve reimplementar a classe pai que possui o method create e show, aonde deve possuir implementacoes especializadas para o create, e o show se mantem generico, criando assim um padrao para apresentacao de telas

A alteração pode ser vista no seguinte [commit](https://github.com/kielsonzinn/billionaires-statistics/commit/1e30a5b0842fac7dafd6b4b1437c83b658df15af)
