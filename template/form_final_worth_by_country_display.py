import plotly.express as px

from template.form_display import FormDisplay


class FormFinalWorthByCountryDisplay(FormDisplay):

    def create_pf(self):
        self._form = px.choropleth(self._data, locations='country', color='finalWorth', scope='world',
                                   color_continuous_scale='Viridis')
        self._form.update_layout(title='Final Worth by Country')
