from flask import render_template, redirect, url_for, request, flash, current_app, session
from app.forms import RegistrationForm, LoginForm
from flask import Blueprint
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import base64
from io import BytesIO
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import classification_report, confusion_matrix
from werkzeug.security import generate_password_hash
import numpy as np
from scipy.interpolate import make_interp_spline
import math
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

main = Blueprint('main', __name__)

# Helper function to check if user is logged in
def login_required(f):
    def wrap(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@main.route('/')
def home():
    return render_template('home.html')

# Register
@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            supabase = current_app.config['SUPABASE_CLIENT']
            user = supabase.table('users').insert({
                "name": form.name.data,
                "email": form.email.data,
                "phone": form.phone.data,
                "password": generate_password_hash(form.password.data)  # Hashing password
            }).execute()
            
            flash('Account created successfully!', 'success')
            return redirect(url_for('main.login'))
        except Exception as e:
            # Menangani kesalahan saat penyimpanan data
            flash(f'An error occurred: {str(e)}', 'danger')
    else:
        # Menampilkan kesalahan validasi
        flash('Please correct the errors in the form.', 'danger')
        print(form.errors)  # Log kesalahan validasi di console

    return render_template('register.html', form=form)


# Fungsi Untuk Login
@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        supabase = current_app.config['SUPABASE_CLIENT']
        response = supabase.table('users').select("*").eq('email', form.email.data).single().execute()
        user_data = response.data
        if user_data and user_data.get('password') == form.password.data:
            session['logged_in'] = True
            session['user_id'] = user_data.get('id')  # Simpan ID pengguna di sesi if perlu
            flash('Login successful!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)

@main.route('/logout')

@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.login'))

@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@main.route('/about')
@login_required
def about():
    return render_template('about.html')

@main.route('/history')
@login_required
def history():
    return render_template('history.html')

def create_plot(x_current, y_current, x_pred, y_pred, sisi, current_distance, predicted_distance):

        plt.figure(figsize=(12, 6))

        # Interpolasi dengan spline untuk membuat kurva lebih halus
        xnew_current = np.linspace(x_current.min(), x_current.max(), 300)
        y_smooth_current = make_interp_spline(x_current, y_current, k=3)(xnew_current)

        xnew_pred = np.linspace(x_pred.min(), x_pred.max(), 300)
        y_smooth_pred = make_interp_spline(x_pred, y_pred, k=3)(xnew_pred)

        # Plot interpolated lines
        plt.plot(xnew_current, y_smooth_current, label='Current', color='blue', linewidth=1)
        plt.plot(xnew_pred, y_smooth_pred, label='After Prediction', color='orange', linewidth=1)

        # Highlight points before prediction (current)
        plt.scatter(x_current, y_current, color='blue', marker='o', edgecolor='black', s=50)
        for x, y in zip(x_current, y_current):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', color='blue')

        # Highlight points after prediction
        plt.scatter(x_pred, y_pred, color='orange', marker='o', edgecolor='orange', s=50)
        for x, y in zip(x_pred, y_pred):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', color='orange')

        # Add early warning markers for negative points
        plt.scatter(x_current[y_current < 0], y_current[y_current < 0], color='red', marker='o', s=100, label='Current Negative')
        plt.scatter(x_pred[y_pred < 0], y_pred[y_pred < 0], color='red', marker='o', s=100, label='Predicted Negative')

        plt.title(f'Grafik {sisi} Over Titik')
        plt.xlabel('Titik')
        plt.ylabel(f'{sisi}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(np.arange(min(x_current.min(), x_pred.min()), max(x_current.max(), x_pred.max()) + 1, 1))

        # Adding annotation for current distance
        plt.text(0.05, 0.95, f'Jarak Sekarang: {current_distance} km', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))

        # Adding annotation for predicted distance
        # plt.text(0.05, 0.90, f'Jarak Prediksi Akan Terjadi Aus: {predicted_distance} km', transform=plt.gca().transAxes,
        #         fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))

        plt.gca().margins(x=0.05, y=0.15)
        plt.tight_layout()

        # Save plot as PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return base64.b64encode(img.getvalue()).decode()
def determine_condition(series):
                    nilai_positif = (series >= 0).sum()
                    nilai_negatif = (series < 0).sum()
                    return 'Bagus' if nilai_positif >= nilai_negatif else 'Aus'

@main.route('/run_test', methods=['GET', 'POST'])
@login_required
def run_test():

    conditions_before = {}
    conditions_after = {}
    predicted_distances = {}
    graphs_urls = {}
    negative_values = {}
    filename = None
    current_distance = None
    assumption_decrease = None
    penurunanAsumsiDalamDesimal = None

    # Tambahkan variabel untuk menyimpan total jarak prediksi
    total_predicted_distance = 0
    jumlah_sisi = 4  # Karena ada 4 sisi
    average_predicted_distance = 0

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        current_distance_str = request.form.get('current_distance', '0')
        diameter_roda_str = request.form.get('diameter_roda', '0')

        try:
            current_distance = float(current_distance_str)
            diameter_roda = float(diameter_roda_str)

            # Rumus Asumsi Penurunan Per Kilometer
            penurunanDiameter = 20 / current_distance
            assumption_decrease = (penurunanDiameter / diameter_roda) * 100
            penurunanAsumsiDalamDesimal = assumption_decrease / 100


        except ValueError:
            flash('Invalid input for distance or wheel diameter.')
            return redirect(request.url)

        filename = file.filename

        

        try:
            if file:
                try:
                    df = pd.read_csv(file)
                except FileNotFoundError:
                    return "File tidak ditemukan. Pastikan file diunggah dengan benar."
                except pd.errors.EmptyDataError:
                    return "File CSV kosong. Mohon unggah file yang valid."
                except pd.errors.ParserError:
                    return "Terjadi kesalahan saat membaca file. Pastikan format CSV sudah benar."

                threshold = 1e-10
                
                # Menggunakan SVR untuk Prediksi Jarak Tempuh
                for sisi in ['Sisi 1', 'Sisi 2', 'Sisi 3', 'Sisi 4']:
                    df_pred = df.copy()
                    predicted_distance = current_distance
                    max_iterations = 10000
                    iteration = 0
                    print('Menghitung', sisi + '....')

                    try:
                        # Persiapan data untuk SVR
                        X = df[['Titik']].values
                        y = df[sisi].values

                        svr_model = SVR(kernel='rbf')
                        svr_model.fit(X, y)  # Melatih model SVR

                        while iteration < max_iterations:
                            # Prediksi nilai dengan SVR
                            df_pred[sisi] = svr_model.predict(X)
                            
                            penurunan_dinamis = (predicted_distance - current_distance) * penurunanAsumsiDalamDesimal
                            df_pred[sisi] = df_pred[sisi] - penurunan_dinamis
                            

                            nilai_positif_setelah_prediksi = (df_pred[sisi] >= 0).sum()
                            nilai_negatif_setelah_prediksi = (df_pred[sisi] < 0).sum()

                            if nilai_negatif_setelah_prediksi > nilai_positif_setelah_prediksi:
                                print(sisi, 'berhasil dihitung')
                                print('---------------------------')
                                print('Df Asli:')
                                print(df[sisi])
                                print('Df Prediksi:')
                                print(df_pred[sisi])

                                break

                            predicted_distance += 1
                            iteration += 1
                            print('Menghitung ' + sisi + ' | Prediksi Jarak = ', predicted_distance, ' Kilometer')

                        # Hitung MSE untuk model SVR
                        mse = mean_squared_error(y, df_pred[sisi])
                        print(f'Mean Squared Error (MSE):', mse)

                        # Membulatkan nilai asli dan prediksi
                        df[sisi] = df[sisi].round(2)
                        df_pred[sisi] = df_pred[sisi].round(2)

                        conditions_before[sisi] = determine_condition(df[sisi])
                        conditions_after[sisi] = 'Aus'

                        predicted_distances[sisi] = predicted_distance
                        negative_values[sisi] = df_pred[df_pred[sisi] < 0].copy()
                        negative_values[sisi]['Titik'] = negative_values[sisi]['Titik'].astype(int)

                        x_current = df['Titik']
                        y_current = df[sisi]
                        x_pred = df_pred['Titik']
                        y_pred = df_pred[sisi]

                        graphs_urls[sisi] = create_plot(x_current, y_current, x_pred, y_pred, sisi, current_distance, predicted_distance)

                        total_predicted_distance += predicted_distance

                    except KeyError as e:
                        return f"Kolom yang diminta tidak ditemukan: {e}. Pastikan semua kolom yang dibutuhkan ada di dataset."
                    except Exception as e:
                        return f"Terjadi kesalahan dalam proses perhitungan atau plotting: {str(e)}"

            # Pastikan total_predicted_distance hanya dihitung jika setidaknya satu prediksi berhasil
            if total_predicted_distance > 0:
                average_predicted_distance = round(total_predicted_distance / jumlah_sisi, 2)

        except Exception as e:
            return f"Terjadi kesalahan yang tidak terduga: {str(e)}"

        # Hitung rata-rata predicted_distance

    # ini untuk return / mengembalikan nilai hasil perhitungan dan prediksi ke website
    return render_template(
        'run_test.html',
        graphs_urls=graphs_urls,
        filename=filename,
        conditions_before=conditions_before,
        conditions_after=conditions_after,
        predicted_distances=predicted_distances,
        current_distance=current_distance,
        assumption_decrease=assumption_decrease,
        negative_values = negative_values,
        average_predicted_distance=average_predicted_distance
    )

    return render_template('run_test.html')


