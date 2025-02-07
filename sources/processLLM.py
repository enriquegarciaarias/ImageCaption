from sources.common.common import logger, processControl, log_

"""
Metadatos:
- Etiqueta de clúster: "Panorámica"
- Nombre del yacimiento: "Santuario de Némesis"
- Título de la imagen: "Planta del Santuario de Némesis"
- Zona del yacimiento: "Ática"

Texto largo: [Inserte aquí el texto de 3000 palabras sobre el yacimiento]

Instrucciones:
Extrae del texto largo una descripción relevante para la imagen, teniendo en cuenta que es una "Panorámica" del "Santuario de Némesis" ubicado en "Ática". La descripción debe ser concisa (máximo 100 palabras) y enfocarse en los elementos visuales o contextuales que podrían aparecer en una imagen panorámica del yacimiento.
Modelo de LLaMA 2 a Utilizar
Modelo recomendado: LLaMA 2 13B (equilibrio entre rendimiento y requisitos computacionales).

Si tienes recursos limitados, puedes usar LLaMA 2 7B.

Si necesitas máxima precisión y tienes recursos suficientes, usa LLaMA 2 70B.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


def huggingface_login():
    try:
        # Add your Hugging Face token here, or retrieve it from environment variables
        token = "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"
        login(token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print("Error logging into Hugging Face:", str(e))
        raise

def processLLM():
    huggingface_login()
    # Cargar el tokenizador y el modelo LLaMA 2
    model_name = "meta-llama/Llama-2-13b-chat-hf"  # Ajusta el tamaño del modelo según tus recursos
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Metadatos y texto largo
    metadata = """
    - Etiqueta de clúster: "Panorámica"
    - Nombre del yacimiento: "Santuario de Némesis"
    - Título de la imagen: "Planta del Santuario de Némesis"
    - Zona del yacimiento: "Ática"
    """

    texto_largo = """
    Acceso
El acceso al yacimiento puede hacerse a través del enlace de Agios Stéfanos de la E-75 cogiendo la dirección a Marathonas. Después de pasar el pueblo de Limiko, la carretera acaba directamente en las puertas del recinto del yacimiento. 

Historia
El nombre del lugar procede de la palabra “ramnos”, un tipo de arbusto que cubre toda la zona.
La Némesis que era adorada en Ramnous era una diosa de la agricultura. Némesis, además, y debido a sus particularidades ctónicas, estaba relacionada en el Ática con los muertos, y en su celebración anual, las Nemeas, tenían lugar combates de antorchas entre los adolescentes que hacían el servicio militar en la fortaleza.
Parece ser que la instauración del culto a Némesis en Ramnous tiene que ver con las Guerras médicas. El lugar fue habitado desde la época protoheládica sin interrupción. Desde los inicios del siglo VI a.C. la cerámica es votiva. De otros fragmentos encontrados se sabe que tenían un uso funerario, una importancia ctónica.
Ctónica era, además, la naturaleza de otra divinidad a la que se rendía culto en el santuario: Temis.
Ramnous alcanzó su apogeo durante los siglos IV y III a.C. Después, poco a poco, el lugar fue abandonándose hasta mediados del siglo IV d.C. Los templos del santuario de Nemea continuaron en pie hasta el final del mencionado siglo.

Mitología
    1. Algunos dicen que cuando Zeus se enamoró de Némesis, ella huyó de él arrojándose al agua y se convirtió en pez, y que él la persiguió transformado en castor y surcando las olas. Ella saltó a tierra y se transformó en diversas fieras, pero no pudo zafarse de Zeus, porque éste tomaba la forma de animales todavía más feroces y rápidos. Por fin, ella se remontó al aire como un ganso silvestre y él se transformó en un cisne y la cubrió triunfalmente en Ramnous. Némesis sacudió sus plumas resignadamente y fue a Esparta, donde Leda, esposa del rey Tindáreo, encontró poco después en un pantano un huevo de color de jacinto que llevó a su casa y ocultó en un cofre; de ese huevo salió Helena de Troya. 
 Otros dicen que Zeus, simulando que era un cisne perseguido por un águila, se refugió en el seno de Némesis y la violó, y que, cuando transcurrió el tiempo debido, ella puso un huevo que Hermes arrojó entre los muslos de Leda cuando estaba sentada en un taburete con las piernas abiertas. Así Leda dio a luz a Helena y Zeus colocó las imágenes del Cisne y el Águila en el firmamento para conmemorar ese ardid.
Sin embargo, el relato más común es que fue con Leda misma con quien se ayuntó Zeus en la forma de un cisne junto al río Eurotas; que ella puso un huevo del que salieron Helena, Castor y Pólux; y que en consecuencia se la deificó como la diosa Némesis. Ahora bien, el marido de Leda, Tindáreo, también se había acostado con ella esa misma noche y, si bien algunos sostienen que los tres eran hijos de Zeus —y también Clitemnestra, quien había salido con Helena, de un segundo huevo—, otros dicen que solamente Helena era hija de Zeus y que Castor y Pólux eran hijos de Tindáreo; otros más afirman que Castor y Clitemnestra eran hijos de Tindáreo, en tanto que Helena y Pólux eran hijos de Zeus.
Según Graves, Némesis era la diosa Luna como Ninfa y, en la forma más antigua del mito de la cacería amorosa, perseguía al rey sagrado a través de sus cambios estacionales de liebre, pez, abeja y ratón —o liebre, pez, pájaro y grano de trigo— y finalmente lo devoraba. Con la victoria del sistema patriarcal la persecución se invirtió: ahora era la diosa la que huía de Zeus. 
Se dice que Leda es una palabra licia (es decir, cretense) que significa «mujer», y Leda era la diosa Latona, o Leto, o Lat, que dio a luz a Apolo y Ártemis en Délos. La fábula de que fue arrojado entre los muslos de Leda puede haberse deducido de una ilustración en la que aparecía la diosa sentada en el banquillo del parto con la cabeza de Apolo saliendo de su útero.
Helena y Helle o Selene son variantes locales de la diosa Luna
Zeus engañó a Némesis, la diosa del culto del cisne en el Peloponeso, apelando a su compasión, exactamente como había engañado a Hera, del culto del cuco cretense. Este mito se refiere, al parecer, a la llegada a ciudades cretenses o pelasgas de guerreros helenos que, para comenzar, rendían homenaje a la Gran Diosa y proporcionaban a sus sacerdotisas maridos obedientes, arrebatándoles luego la soberanía suprema.

El yacimiento
La primera excavación científica que tuvo lugar en el santuario tuvo lugar en el 1813 a cargo del arquitecto inglés John Peter Gandy Deering, miembro de la expedición de los Diletantes.
El 1880 Δημήτριος Φίλιος excava en la fortaleza y en el recinto funerario de Ιεροκλέους. Después de 10 años, Valerio Stays, saca a la luz casi todo lo que es visible hoy en día: El santuario de Némesis, el santuario de Amfiaraos, el interior de la fortaleza y muchos recintos funerarios.
En 1975 volvieron a comenzar las excavaciones sobre una base totalmente nueva. El objetivo no era ya el descubrimiento de estatuas y epigrafías sino el restablecimiento de la imagen real de Ramnous, imagen que nada tenía que ver con la deducida durante años por los hallazgos que proporcionaban las excavaciones. 
Los lugares principales de Ramnous son la vía norte – sur (vía sagrada que comunicaba la fortaleza con el demos cercano de Tricorito) los santuarios de Némesis y Anfiarao y la fortaleza, ocupada en una gran parte por edificios públicos y privados. 
A lo largo de la vía sagrada se encontraban monumentos funerarios, los más importantes de los cuales serán tratados a continuación. En el tramo sur de la vía son muy importantes los recintos funerarios de Menéstides y Eufránoros.
El recinto funerario de la familia de Menéstides (4), construido de mármol blanco, tiene una longitud de 7,5 m y una altura de 1,7 m. Está decorado por un relieve datado del 380 – 370 a.C. único por su tamaño. Era obra de algún famoso escultor ateniense.
El vecino recinto funerario de Eufránoros (5), del 330 a.C., construido de mármol blanco local, tiene una longitud de 7,9 m y una altura de 3,6 m. A pesar de la destrucción que ha soportado, se conserva su gran estela funeraria que refiere, cronológicamente, los miembros de la familia que están aquí enterrados. En la base de la estela hay grabado un interesante epigrama. 
Más adelante nos encontramos con el santuario de Némesis (9). La zona alrededor del templo no estaba, como hoy, desierta. Casas, parcelas, construcciones agrícolas, pozos, vallas y almacenes daban otro aspecto al lugar.
La imagen actual del lugar no da una idea de lo que existió aquí en un tiempo pasado. En cualquier caso, muy cerca de los templos se conservan los restos de muchas casas y solares que muestran como los templos y el santuario se encontraban en el centro de un pequeño complejo muy visitado, mientras que en los alrededores y en las laderas había casas aisladas.
Al SO del santuario y muy cerca del recinto de Eufránoros, se han descubierto las ruinas de una casa oeste (7) con grandes patios para las tareas domésticas o para el estabulado de los animales, un gran espacio con una stoa enfrente y habitaciones para la permanencia de los huéspedes o el almacenamiento de las cosechas del cereal.
En el punto donde se encuentra la casa acaba la vía sur. Su último tramo estaba empedrado y en su lado oeste había ανάλημμα frente a la cual había ofrendas institucionales. Desde el espacio del santuario se ve el golfo euboico y la isla de Eubea. 
El santuario se construyó sobre una terraza artificial sostenida por bloques isodómicos al norte y al este en el siglo VI a.C., antes de construirse el gran templo. La excavación de la terraza proporcionó cerámica de todas las épocas, la cual llega cronológicamente hasta las primeras décadas del siglo V a.C. 
El templo más antiguo del santuario está datado en las primeras décadas del siglo VI a.C. A este templo, del cual solo se conservan algunas tejas de la cubierta, le sucedió un pequeño templo de finales del siglo VI a.C. de piedra caliza, de estilo dórico con dos columnas en la fachada entre dos hastiales, el cual se construyó en el mismo sitio que el anterior. El templo, que estaba hecho de piedra caliza, fue destruido probablemente en el 480 – 479 a.C. por los persas.
En el santuario, el visitante puede ver las ruinas de dos templos. El más antiguo es el que se encuentra situado al sur, construido a principios del siglo V a.C. con estilo lesbio y de dimensiones 9,9 m de largo por 6,15 m de ancho. 
La fachada del templo pequeño era sencilla, un muro con una puerta en el centro. En la parte superior de la fachada había un frontón del cual se conserva una parte del tímpano. La cubierta del templo se realizaba a base de tejas de estilo corintio. Estaba compuesto por una sala rectangular y un vestíbulo. 
El pequeño templo se conservó hasta el siglo IV d.C. como tesoro y almacén. En el interior de su cella se encontraron importantes estatuas que se encuentran hoy en día en el Museo Nacional de Atenas. Entre ellas, la estatua de mármol pario de la diosa Temis del escultor de Ramnous, Queréstrato (finales del siglo IV a.C.)
El último templo, cronológicamente hablando, es el conocido como de Ramnous. Se considera obra del llamado arquitecto del Teseion. Construido en la segunda mitad del siglo V a.C., dentro del programa de construcciones en el Ática impulsado por Pericles, tenía una longitud de 21,4 m y una anchura de 10,05 m con una perístasis de 6 x 12 columnas dóricas, una naos, una pronaos dístila y un opistodomo in antis.
El templo se mantuvo inconcluso, con columnas no estriadas y los escalones del estilóbato inacabados. Las tejas del techo eran de mármol pentélico mientras que el estilóbato era de mármol local más azulado. Este edificio fue el que estuvo precedido por los dos templos del siglo VI a.C. de los que hemos hablado.
A las metopas y los frontones les faltaba la decoración escultórica. Sin embargo, las cumbres de los frontones estaban adornadas con composiciones escultóricas y las esquinas con Quimeras. El friso, sin embargo, estaba totalmente finalizado. 
Al fondo de la cella se encontraba la estatua de Némesis de mármol de Paros, obra de Agorákrito de Paros, un discípulo de Fidias, y, frente a ella, la mesa de ofrendas. De la estatua, de la que se han conservado cientos de trozos, conocemos la forma gracias a las investigaciones de los últimos años. La estatua se erigía sobre un pedestal de mármol. Sostenía una copa decorada con figuras de etíopes en la mano derecha y una rama de manzano en la izquierda. Una corona decorada con ciervos y pequeñas Victorias descansaba en su cabeza. Un gran fragmento de la cabeza se conserva en el Museo Británico.
La restauración de la forma del pedestal ha constituido uno de los grandes problemas de la arqueología clásica durante los últimos 100 años. El visitante puede ver el pedestal de la estatua de culto, que se conserva desde el siglo V a.C., en una sala (41) especial. La decoración está hecha a base de relieves de Leda y Helena junto a Némesis. También, según Pausanias, estaban representados Tindáreo, Agamenón, Menelao y Pirro, hijo de Aquiles.  En esta misma sala, que se encuentra al este del santuario de Némesis, se ha reconstruido el entablamento y el frontón occidental del gran templo de Némesis, el gran templete de Διογείτονος (segunda mitad del siglo IV a.C.), el de Ιεροκλέους, el relieve del recinto funerario de Menéstides y el templete del recinto de Πυθάρχος.
Al este del gran templo se conservan los cimientos del altar (3,25 x 7,8 m), del cual no tenemos ningún otro dato.
En la parte norte del santuario había, además, dos construcciones: una stoa de 34 m de longitud, con columnas de madera en la fachada, que miraba hacia el norte, y, enfrente de la stoa, una fuente con un pórtico de dos columnas. Las dos habían sido construidas durante el siglo V a. C. La fuente, antes de la construcción del gran templo y del ανάλημμα y, la stoa, después del ανάλημμα.
El agua de la fuente venía de una profunda cisterna tallada en el mármol del sustrato del santuario. Esta cisterna se llenaba por medio de un acueducto que recogía las aguas de lluvia provenientes de los tejados de los edificios de alrededor y de los correspondientes al gran y pequeño templo.
Un edificio rectangular helenístico (10) se alza frente al santuario, al otro lado de la vía sagrada. Se cree que se trata de la sede de los comandantes del demos, aunque también pudiera ser un edificio relacionado con las actividades del santuario. 
Dejando el santuario de Nemea, continuamos nuestro camino hacia la fortaleza. A este tramo de la vía acompañan extraordinarios monumentos funerarios. El primero es el recinto de Diogitón (¿?) de mármol local blanco de 6,5 m de largo x 4,16 m de alto, de finales del siglo IV a.C. Ha sido restaurado recientemente. Estaba adornado con una gran estela en la que estaban grabados los nombres de los muertos. Al lado de la estela había dos templetes. El mayor de ellos, con dos columnas jónicas en la fachada y dos estatuas de mujer en el interior, constituye uno de los últimos y más extraordinarios monumentos de su tipo en el Ática.
Merece la pena señalar también las epigrafías que se grabaron en su fachada durante el siglo III a.C. relativas a las personas que fueron enterradas aquí.
Continuando nuestra calle nos encontramos de frente y a la derecha el pequeño recinto funerario de Mnisikratias (12), de mármol blanco local del siglo IV a.C., y más abajo, a la izquierda el recinto funerario de Pitharco (15), de la misma época, hecho de piedra de poros, que estaban adornados con un templete y dos estelas, una grande con los nombres de los muertos, y una pequeña con una representación en relieve. 
A su lado está, también de piedra de poros, el recinto funerario de Fanócrates (15), del cual se conserva la estela con los nombres de los muertos y otra con un relieve de un muchacho y su pequeño esclavo.
Después del recinto funerario de Fanócrates, a la izquierda de la vía vemos las imponentes ruinas del gran recinto funerario de Ierokleos (16), de piedra de poros. A este recinto, que fue saqueado por los ladrones de tesoros en el siglo XIX, pertenecen famosas obras escultóricas como el relieve que representa a Ierokles, uno de sus cinco hijos, con su prometida Demostrata. Otro relieve famoso del recinto es el que representa a otro de los hijos de Ierokleos, Iéronos, con su mujer Lisipe. Este conjunto escultórico, uno de los más bellos del Ática, se encuentra en el Museo Arqueológico Nacional de Atenas. Sin embargo, el frontón del templete con los nombres de los muertos y su pedestal se encuentran aquí.
Importantes recintos que veremos todavía en nuestro camino son el reconstruido recinto funerario de los hermanos Atinodoro y Dromocles (18) con tres altas estelas y otro recinto funerario anónimo (21) con un templete y vasos de mármol en el lado oeste de la vía, ambos de la segunda mitad del siglo IV a.C.
La fortaleza (25) de Ramnous está construida sobre una colina rodeada por fuertes murallas de 800 m de longitud construidas con grandes bloques de mármol.
Su puerta principal (26), que se encontraba en el lado sur, estaba protegida por dos torres ortogonales. Torres también había, cada cierto espacio, en los otros lados de la muralla.
La fortaleza se divide en dos partes: la parte de arriba, en donde se encontraban las instalaciones militares, y la parte de abajo en donde se encontraban los edificios públicos y privados.
La fortaleza tomó su primitiva forma en las últimas décadas del siglo V a.C. y su misión era asegurar a los atenienses la libre circulación de barcos en el golfo euboico y el transporte sin obstáculos del grano desde Eubea hasta Atenas. Particular importancia obtuvo durante los tiempos de Aristóteles en los que los jóvenes, en su segundo año de servicio militar, realizaban patrullas por el Ática teniendo como campamentos bases este tipo de fortalezas. La de Ramnous era una de las más importantes.
Con la muerte de Alejandro, estas patrullas desaparecieron y los macedonios, ocupando las fortalezas atenienses, se aseguraban la sumisión de Atenas.
Entrando por la puerta principal nos encontramos primeramente una casa con una torre (34) redonda y un patio. Al norte de ella había un gimnasio (35) de unos 1.200 m2 que tenía unos propileos monumentales en su lado sur con un dintel de una sola pieza.
Al norte del gimnasio se encuentra el extraño, por su forma tan simple, teatro (36) de Ramnous. Como cávea utilizaba la ladera de la colina y como orquestra la superficie plana que había frente a los asientos de la presidencia, los cuales constituyen el único elemento arquitectónico que se ha conservado del teatro.
El teatro y el espacio de alrededor se utilizaban también como ágora.
Alrededor del gimnasio y del teatro se encuentran las ruinas de viviendas y parcelas. Aquí se encontraba el santuario de Dionisos y del héroe Argiguetos, así como los edificios públicos.
En el punto más alto de la colina, en la parte de arriba de la fortaleza, estaban las instalaciones militares. Estaban separadas de la parte baja de la fortaleza mediante un muro con una puerta. Atravesamos la puerta, que estaba protegida por una torre, y llegamos a la acrópolis (37). Tenemos, sin embargo, una vista excepcional de los dos puertos: el puerto este (31), cubierto totalmente por la vegetación y los vertidos de tierra, y el puerto oeste (39) con la iglesita de Santa Marina. La importancia de los puertos para los habitantes de Ramnous era evidente. Aquí estaban atracados los barcos de guerra que patrullaban el golfo euboico y los barcos mercantes que comerciaban con las islas de alrededor.
El pequeño santuario de Anfiarao (24) se encuentra al SO de la puerta principal de entrada a la fortaleza, encima de las rocas de la colina.  Quedan muy pocas ruinas y no es posible hacerse una idea clara de la forma del santuario. Sabemos por una epigrafía que una parte de él, el templo, estaba techado con tejas de cerámica y que en su interior estaba la mesa sagrada y la estatua de culto del dios, mientras que fuera estaba el altar. Había también una stoa. 
Con este conjunto de datos tenemos que identificar las dos habitaciones del santuario que se han conservado. La occidental sería el templo y la oriental la stoa que se utilizaría, además, como ενκοιμητήριο para los enfermos. Entre las dos hay un patio en donde estaba el altar.  En su lado sur había ofrendas votivas alineadas, esculturas y relieves, y frente a todo ello, un poyete.
El Anfiareio era, en un principio, un santuario medicinal y ctónico del héroe médico Aristómaco, como sabemos por las epigrafías y las fuentes filológicas. Durante el siglo IV a.C. esta divinidad se asimiló al famosísimo Anfiarao, al que adoraban en Oropós. Quizás, el Amfiareio se utilizase como hospital para los heridos de la fortaleza.
    """

    # Crear el prompt
    prompt = f"""
    Metadatos:
    {metadata}
    
    Texto largo:
    {texto_largo}
    
    Instrucciones:
    Extrae del texto largo una descripción relevante para la imagen, teniendo en cuenta que es una "Panorámica" del "Santuario de Némesis" ubicado en "Ática". La descripción debe ser concisa (máximo 100 palabras) y enfocarse en los elementos visuales o contextuales que podrían aparecer en una imagen panorámica del yacimiento.
    """

    # Tokenizar y generar la descripción
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=300)  # Ajusta max_length según sea necesario
    descripcion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Descripción generada:", descripcion)