# app/routes.py
from flask import render_template, redirect, url_for, request, flash, current_app
from app.forms import RegistrationForm, LoginForm
from flask import Blueprint
from flask import session, redirect, url_for, flash, Blueprint
import csv
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter  # Tambahkan impor ini
import numpy as np

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        supabase = current_app.config['SUPABASE_CLIENT']
        user = supabase.table('users').insert({
            "name": form.name.data,
            "email": form.email.data,
            "phone": form.phone.data,
            "password": form.password.data
        }).execute()
        flash('Account created successfully!', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html', form=form)

@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        supabase = current_app.config['SUPABASE_CLIENT']
        response = supabase.table('users').select("*").eq('email', form.email.data).single().execute()
        user_data = response.data
        if user_data and user_data.get('password') == form.password.data:
            flash('Login successful!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)

@main.route('/logout')
def logout():
    # Hapus semua data dalam sesi
    session.clear()
    # Berikan pesan kepada pengguna bahwa mereka telah logout
    flash('You have been logged out.', 'info')
    # Arahkan kembali ke halaman login
    return redirect(url_for('main.login'))

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/history')
def history():
    return render_template('history.html')


@main.route('/run_test', methods=['GET', 'POST'])
def run_test():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = file.filename
        
        if file:
            df = pd.read_csv(file, delimiter=',')
            df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d_%H:%M:%S.%f')
            
            # Algorimta KNN Utk Keputusan
            df['Master PV'] = df['Master PV'].apply(lambda x: float(str(x).replace(',', '.')))
            df['Condition'] = df['Master PV'].apply(lambda x: 'Aus' if x < 0 else 'Bagus')

            # Prepare features and labels
            X = df[['Master PV']]
            y = df['Condition']

            # Create training data with some sample points
            training_data = pd.DataFrame({
                'Master PV': np.concatenate([df['Master PV'].values, [df['Master PV'].mean()]]),
                'Condition': df['Condition'].tolist() + ['Bagus']
            })
            print(training_data)

            # Train the KNN model
            X_train = training_data[['Master PV']]
            y_train = training_data['Condition']
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            knn = KNeighborsClassifier(n_neighbors=1)  # Use 1 neighbor due to limited data
            knn.fit(X_train_scaled, y_train)
            
            # Make prediction on the provided data
            X_scaled = scaler.transform(X)
            prediction = knn.predict(X_scaled)
            predicted_condition = 'Aus' if prediction[0] == 'Aus' else 'Bagus'

            # Kode Tampilan Grafik
            plt.figure(figsize=(10,6))
            ax = sns.lineplot(x=df['Time'], y=df['Master PV'], marker='o')

            for i, (x, y) in enumerate(zip(df['Time'], df['Master PV'])):
                if i % 2 == 0:  # Annotate every 2nd point to reduce clutter
                    ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

            plt.title('Master PV Over Time')
            plt.xlabel('Waktu')
            plt.ylabel('Master PV')

            formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
            ax.yaxis.set_major_formatter(formatter)
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            ax.grid(True, linestyle='--', alpha=0.7)

            # Save the plot as a PNG image
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Generate table data
            table_data = df.to_html(classes='table-auto w-fit text-end text-lg border-collapse border border-gray-300', index=False, header=True)

            return render_template('run_test.html', plot_url=plot_url, condition=predicted_condition, table_data=table_data, filename=filename)
    
    return render_template('run_test.html')
