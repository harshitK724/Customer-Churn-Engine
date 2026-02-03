from src.data.schema import CustomerFeatures


def test_valid_customer():
    customer = CustomerFeatures(
        customer_id="7590-VHVEG",
        gender="Female",
        senior_citizen=0,
        partner="Yes",
        dependents="No",
        tenure=1,
        phone_service="No",
        multiple_lines="No phone service",
        internet_service="DSL",
        online_security="No",
        online_backup="Yes",
        device_protection="No",
        tech_support="No",
        streaming_tv="No",
        streaming_movies="No",
        contract="Month-to-month",
        paperless_billing="Yes",
        payment_method="Electronic check",
        monthly_charges=29.85,
        total_charges=29.85,
    )
    assert customer.tenure == 1
