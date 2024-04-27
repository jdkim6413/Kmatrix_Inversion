
class UserInput:
    def __init__(self, gen_matrix=True, matrix_size = 8, n_matrix = 1_000, drive_path='./',
                 is_crawling=True, is_modelfromfile=False, is_tune_para=True):
        self.gen_matrix = gen_matrix
        self.matrix_size = matrix_size
        self.n_matrix = n_matrix

        self.is_crawling = is_crawling
        self.is_modelfromfile = is_modelfromfile
        self.is_tune_para = is_tune_para

        # 파일 기본경로 지정
        self.drive_path = drive_path
        self.data_path = self.drive_path + 'data/'
        self.output_path = self.drive_path + 'output/'
