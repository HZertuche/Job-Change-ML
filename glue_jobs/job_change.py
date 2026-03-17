def MyTransform (glueContext, dfc) -> DynamicFrameCollection:
    from awsglue.dynamicframe import DynamicFrame, DynamicFrameCollection
    from pyspark.sql.functions import col, when, regexp_replace
    from pyspark.sql.window import Window
    
    # Obtener DataFrame
    df = dfc.select(list(dfc.keys())[0]).toDF()
    
    # Identifying cities
    df = df.withColumn(
        "city",
        regexp_replace("city","city_","").cast("int")
)
    
    # Limpiar columna gender
    df = df.withColumn(
        "gender",
        when(col("gender") == "Male", 1)
        .when(col("gender") == "Female", 2)
        .otherwise(0)
    )
    
    # Limpiar columna experience
    df = df.withColumn(
        "experience",
        when(col("experience") == ">20", 21)
        .when(col("experience") == "<1", 0)
        .otherwise(col("experience").cast("int"))
    )
    
    #Limpiar columna Relevant Experiemce
    df = df.withColumn(
        "relevent_experience",
        when(col("relevent_experience") == "Has relevent experience", 1)
        .otherwise(0)
    )
    # Limpiar columna Enrolled University
    df = df.withColumn(
        "enrolled_university",
        when(col("enrolled_university") == "no_enrollment", 0)
        .when(col("enrolled_university") == "Full time course", 1)
        .when(col("enrolled_university") == "Part time course", 2)
        .otherwise(3)
    )
    # Limpiar columna Company type
    df = df.withColumn(
        "company_type",
        when(col("company_type") == "Pvt Ltd", 1)
        .when(col("company_type") == "Funded Startup", 2)
        .when(col("company_type") == "Public Sector", 3)
        .when(col("company_type") == "NGO", 4)
        .when(col("company_type") == "Early Stage Startup", 5)
        .otherwise(0)
    ) 
    
    
    # Limpiar last_new_job
    df = df.withColumn(
        "last_new_job",
        when(col("last_new_job") == ">4", 5)
        .when(col("last_new_job") == "never", 0)
        .otherwise(col("last_new_job").cast("int"))
    )
    
    # Convertir company_size a categorías numéricas
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
    
    # Si es string como '0.0' o '1.0'
    df = df.withColumn("target", col("target").cast("float").cast("int"))
    
    # Convertir a categorias los niveles de educacion
    df = df.withColumn(
        "education_level",
        when(col("education_level") == "Primary School", 1)
        .when(col("education_level") == "High School", 2)
        .when(col("education_level") == "Graduate", 3)
        .when(col("education_level") == "Masters", 4)
        .when(col("education_level") == "Phd", 5)
        .otherwise(0)
    )
    
    
    # Detectar perfil senior
    df = df.withColumn(
        "senior_candidate",
        when(col("experience") >= 10, 1).otherwise(0)
    )
    
    # Convertir nuevamente a DynamicFrame
    dynamic_frame = DynamicFrame.fromDF(df, glueContext, "cleaned_data")
    
    return DynamicFrameCollection({"CustomTransform0": dynamic_frame}, glueContext)