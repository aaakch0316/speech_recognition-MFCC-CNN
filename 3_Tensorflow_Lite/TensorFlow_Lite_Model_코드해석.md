```
from tensorflow import lite
from tensorflow.keras import models

\# Parameters
keras_model_filename = 'wake_word_stop_model.h5'
tflite_filename = 'wake_word_stop_lite.tflite'

\# Convert model to TF KLite model
model = models.load_model(keras_model_filename)
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_filename, 'wb').write(tflite_model)
```

##### load_model

```
tf.keras.models.load_model(
    filepath, custom_objects=None, compile=True, options=None
)
```

| Arguments(인수) | 설명                                                         |
| --------------- | ------------------------------------------------------------ |
| filepath        | 아래의 2가지중 하나이다. <br />- 저장된 모델로 가는 길(string이나 pathlib.Path object) |
| custom_objects  | Optional] deserialization(직열화)가 이루어 지는 동안 고려되는 함수나 커스텀 틀래스에 대한, strings으로 된 사전적 mapping 이름이다. |
| compile         | Boolean,  로딩 이후의 모델을 컴파일 할지에 대한 Boolean이다. |
| options         | Optional] 저장된 모델에서 로드하기 위한 옵션을 지정하는 선택적인tf.saved_model.LoadOptions 객체이다. |

`Returns` - A Keras model instance. If the original model was compiled, and saved with the optimizer, then the returned model will be compiled. Otherwise, the model will be left uncompiled. In the case that an uncompiled model is returned, a warning is displayed if the `compile` argument is set to `True`.

