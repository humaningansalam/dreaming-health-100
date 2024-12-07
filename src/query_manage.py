# query_builder.py
import psycopg2
import streamlit as st

def build_prescription_query(age, height, weight, additional_measurements):
    """
    사용자 입력을 바탕으로 운동 처방을 위한 SQL 쿼리를 동적으로 생성합니다.

    Parameters:
        age (int): 사용자 나이
        height (int): 사용자 키 (cm)
        weight (int): 사용자 몸무게 (kg)
        additional_measurements (dict): 추가 측정항목의 딕셔너리 (키: 컬럼명, 값: 입력값)

    Returns:
        tuple: 최종 쿼리 문자열과 파라미터 리스트
    """
    base_query = """
    SELECT mvm_prscrptn_cn
    FROM users_measurements
    """
    
    # 기본 항목들
    order_by_clauses = [
        "ABS(mesure_age_co - %s)",
        "ABS(CAST(mesure_iem_001_value AS DECIMAL) - %s)",  # 키
        "ABS(CAST(mesure_iem_002_value AS DECIMAL) - %s)"   # 몸무게
    ]
    params = [age, height, weight]
    
    # 입력된 추가 항목 값들에 대해 쿼리 반영
    for column_name, user_value in additional_measurements.items():
        if user_value:
            try:
                order_by_clauses.append(f"ABS(CAST({column_name} AS DECIMAL) - %s)")
                params.append(float(user_value))  # 사용자 입력값 추가
            except ValueError:
                # 숫자로 변환할 수 없는 값이 입력된 경우 예외 처리
                pass
    
    # ORDER BY 절 완성
    order_by_clause = " + ".join(order_by_clauses)
    final_query = f"""
    {base_query}
    ORDER BY {order_by_clause}
    LIMIT 5
    """

    return final_query, params



def fetch_similar_prescriptions(age, height, weight, additional_measurements):
    """
    데이터베이스에서 유사한 운동 처방을 검색합니다.

    Parameters:
        age (int): 사용자 나이
        height (int): 사용자 키 (cm)
        weight (int): 사용자 몸무게 (kg)
        additional_measurements (dict): 추가 측정항목의 딕셔너리

    Returns:
        list: 유사한 운동 처방 결과 리스트
    """
    # DB 연결
    conn = psycopg2.connect(
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PWD"],
        host=st.secrets["DB_URL"],
        port=st.secrets["DB_PORT"]
    )
    cursor = conn.cursor()

    # 동적 쿼리 생성
    final_query, params = build_prescription_query(age, height, weight, additional_measurements)
    
    # 쿼리 실행
    cursor.execute(final_query, tuple(params))
    results = cursor.fetchall()

    # 연결 종료
    cursor.close()
    conn.close()

    return results