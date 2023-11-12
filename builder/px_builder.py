import plotly.express as px


class PxBuilder:
    __dimisions: None
    __data: None
    __title: None

    def with_dimensions(self, dimensions):
        self.__dimisions = dimensions
        return self

    def with_data(self, data):
        self.__data = data
        return self

    def with_title(self, title):
        self.__data = title
        return self

    def build(self):
        fig = px.parallel_coordinates(self.__data, dimensions=self.__dimisions)
        fig.update_layout(title=self.__title)

        return fig
