import csv
from django.core.management.base import BaseCommand
from myapps.models import WindData, WeatherData

class Command(BaseCommand):
    help = 'Import weather data from CSV'

    def handle(self, *args, **kwargs):
        with open('2.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                try:
                    wind_data_id = int(row['id'])  # Assuming 'id' is in your CSV headers
                    wind_data = WindData.objects.get(id=wind_data_id)
                    
                    WeatherData.objects.create(
                        formatted_date=row['Formatted Date'],
                        summary=row['Summary'],
                        precip_type=row.get('Precip Type', 'unknown'),
                        temperature=float(row['Temperature (C)']),
                        apparent_temperature=float(row['Apparent Temperature (C)']),
                        humidity=float(row['Humidity']),
                        pressure=float(row['Pressure (millibars)']),
                        daily_summary=row['Daily Summary'],
                        wind=wind_data
                    )
                except WindData.DoesNotExist:
                    print(f"No WindData found for id: {wind_data_id}")
                except Exception as e:
                    print(f"Error creating WeatherData: {e}")

        self.stdout.write(self.style.SUCCESS('Successfully imported weather data'))
