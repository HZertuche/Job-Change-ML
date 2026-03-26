def MyTransform (glueContext, dfc) -> DynamicFrameCollection:
    from awsglue.dynamicframe import DynamicFrame, DynamicFrameCollection
    from pyspark.sql.functions import col, when, regexp_replace
    from pyspark.sql.window import Window
    
    # Load the input DynamicFrame as a Spark DataFrame
    df = dfc.select(list(dfc.keys())[0]).toDF()
    
    # Extract the numeric city identifier from the city column
    df = df.withColumn(
        "city",
        regexp_replace("city","city_","").cast("int")
)
    # Encode gender values as numeric categories
    df = df.withColumn(
        "gender",
        when(col("gender") == "Male", 1)
        .when(col("gender") == "Female", 2)
        .otherwise(0)
    )
    
    # Standardize experience values and convert them to integers
    df = df.withColumn(
        "experience",
        when(col("experience") == ">20", 21)
        .when(col("experience") == "<1", 0)
        .otherwise(col("experience").cast("int"))
    )
    
    # Encode relevant experience as a binary feature
    df = df.withColumn(
        "relevent_experience",
        when(col("relevent_experience") == "Has relevent experience", 1)
        .otherwise(0)
    )
    # Encode university enrollment status as numeric categories
    df = df.withColumn(
        "enrolled_university",
        when(col("enrolled_university") == "no_enrollment", 0)
        .when(col("enrolled_university") == "Full time course", 1)
        .when(col("enrolled_university") == "Part time course", 2)
        .otherwise(3)
    )
    # Encode company type as numeric categories
    df = df.withColumn(
        "company_type",
        when(col("company_type") == "Pvt Ltd", 1)
        .when(col("company_type") == "Funded Startup", 2)
        .when(col("company_type") == "Public Sector", 3)
        .when(col("company_type") == "NGO", 4)
        .when(col("company_type") == "Early Stage Startup", 5)
        .otherwise(0)
    ) 
    
    # Standardize last_new_job values and convert them to integers
    df = df.withColumn(
        "last_new_job",
        when(col("last_new_job") == ">4", 5)
        .when(col("last_new_job") == "never", 0)
        .otherwise(col("last_new_job").cast("int"))
    )
    
    # Encode company size as ordinal numeric categories
    df = df.withColumn(
        "company_size",
        when(col("company_size") == "<10", 1)
        .when(col("company_size") == "10-49", 2)
        .when(col("company_size") == "50-99", 3)
        .when(col("company_size") == "100-499", 4)
        .when(col("company_size") == "500-999", 5)
        .when(col("company_size") == "1000-4999", 6)
        .when(col("company_size") == "5000-9999", 7)
        .when(col("company_size") == "10000+", 8)
        .otherwise(0)
    )
    
    # Cast target values from string/float format to integer
    df = df.withColumn("target", col("target").cast("float").cast("int"))
    
    # Encode education level as ordinal numeric categories
    df = df.withColumn(
        "education_level",
        when(col("education_level") == "Primary School", 1)
        .when(col("education_level") == "High School", 2)
        .when(col("education_level") == "Graduate", 3)
        .when(col("education_level") == "Masters", 4)
        .when(col("education_level") == "Phd", 5)
        .otherwise(0)
    )
    
    # Create a binary flag to identify senior candidates
    df = df.withColumn(
        "senior_candidate",
        when(col("experience") >= 10, 1).otherwise(0)
    )
    
    # Convert the cleaned DataFrame back to a DynamicFrame
    dynamic_frame = DynamicFrame.fromDF(df, glueContext, "cleaned_data")
    
    return DynamicFrameCollection({"CustomTransform0": dynamic_frame}, glueContext)