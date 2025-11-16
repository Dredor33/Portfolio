# Datos_para_tablas 📂

Esta carpeta contiene los archivos de datos generados a partir del Notebook de ETL, listos para **poblar la base de datos en MySQL**.

La estructura refleja el ciclo de vida de Pontia Logista con las tablas de hechos:  
1. **Compras** → información sobre fecha de compra, precio y proveedor  
2. **Productos** → detalles de peso y tipo de fruta  
3. **Ventas** → información sobre fecha de venta, precio y tienda  

El resto de archivos corresponden a dimensiones para aligerar el peso del esquema.  

Con este modelo relacional se consigue que muchas consultas solo requieran de una o dos tablas de hechos, mejorando la eficiencia en las consultas a realizar.
