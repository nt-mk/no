import gdown
import tensorflow as tf
import streamlit as st

# تحميل النموذج من Google Drive باستخدام gdown
file_url = 'https://drive.google.com/uc?id=1ryk6CX8b_GaKHXAmf-nabhXT0ypXKFOU'
output = 'vgg19_the_last_model.keras'

# تحميل النموذج
gdown.download(file_url, output, quiet=False)

# تحميل النموذج باستخدام Keras
model = tf.keras.models.load_model(output)

# عرض واجهة Streamlit
st.title('نموذج VGG19 المحمّل')

# عرض ملخص للنموذج
st.subheader('ملخص النموذج:')
st.text(model.summary())

# إضافة صورة لاختبار النموذج
uploaded_file = st.file_uploader("قم بتحميل صورة لاختبار النموذج", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # تحميل الصورة
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array /= 255.0  # تطبيع الصورة

    # التنبؤ باستخدام النموذج
    predictions = model.predict(img_array)
    st.image(uploaded_file, caption='الصورة التي تم تحميلها.', use_column_width=True)
    st.write(f"التنبؤ: {predictions}")
