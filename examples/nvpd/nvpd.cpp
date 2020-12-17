#include <cstdio>
#include <celerity.h>

int len_str1 = 100, len_str2 = 100;
const int ALPHASIzE = 4;
const char AlphaBATES[] = "acgt";


template <typename T>
void init_first_row(celerity::distr_queue& queue, celerity::buffer<T, 1> buf) {
    queue.submit([=](celerity::handler& cgh) {
        auto dw_buf = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
        cgh.parallel_for<class init_first_row>(buf.get_range(), [=](cl::sycl::item<1> item) { dw_buf[item[0]] = item[0]; });
        });
}

void compute_ith_row(celerity::distr_queue queue,
    celerity::buffer<int, 1> row_pre_buf,
    celerity::buffer<int, 1> row_curr_buf,
    celerity::buffer<int, 2> mi_table_buf,
    celerity::buffer<unsigned char, 1> str2_buf,
    const unsigned char ch, const int ci, const int i, const int iter) {

    queue.submit([=](celerity::handler& cgh) {
        auto a = row_curr_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
        auto b = row_pre_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<1>());
        auto s2 = str2_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
        auto mit = mi_table_buf.get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::all<2>());
        cgh.parallel_for<class compute_ith_row>(cl::sycl::range<1>(iter), [=](cl::sycl::item<1> item) {
            int j = item[0];
            int val = 0;
            if (j == 0)
            {
                a[j] = i;
            }
            else if (ch == s2[j])
            {
                val = b[j - 1];
                a[j] = val;
            }
            else
            {
                int indx = mit[cl::sycl::id<2>(ci, j)];
                int x;
                int y = b[j] + 1;
                int z = b[j - 1] + 1;

                if (indx != 0)
                {
                    x = (j - indx) + (b[indx - 1]);
                    if (x <= y && x <= z)
                    {
                        a[j] = val = x;
                    }
                    else if (y <= x && y <= z)
                    {
                        a[j] = val = y;
                    }
                    else
                    {
                        a[j] = val = z;
                    }
                }
                else
                {
                    if (y <= z)
                    {
                        a[j] = val = y;
                    }
                    else
                    {
                        a[j] = val = z;
                    }
                }
            }
            });
        });
}


int main(int argc, char** argv)
{
    celerity::runtime::init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int numProc;
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);
    const bool is_master = rank == 0;

    if (argc < 4)
    {
        if (is_master)
            printf("Please provide arguements [filename1] [length1] [filename2] [length2]\n");
        return EXIT_FAILURE;
    }


    char* fname1 = argv[1];
    len_str1 = atoi(argv[2]);
    char* fname2 = argv[3];
    len_str2 = atoi(argv[4]);



    unsigned char* str1 = (unsigned char*)malloc((len_str1 + 2) * sizeof(unsigned char));
    unsigned char* str2 = (unsigned char*)malloc((len_str2 + 2) * sizeof(unsigned char));

    FILE* fp1, * fp2;
    fp1 = fopen(fname1, "r");
    if (!fp1)
    {
        printf("Couldn't open file %s\n", fname1);
        return EXIT_FAILURE;
    }

    fp2 = fopen(fname2, "r");
    if (!fp2)
    {
        fclose(fp1);
        printf("Couldn't open file %s\n", fname2);
        return EXIT_FAILURE;
    }


    str1[0] = str2[0] = ' ';
    for (int iu = 1; iu <= len_str1; iu++)
    {
        str1[iu] = (unsigned char)fgetc(fp1);
    }

    for (int iu = 1; iu <= len_str2; iu++)
    {
        str2[iu] = (unsigned char)fgetc(fp2);
    }

    fclose(fp1);
    fclose(fp2);


    str1[len_str1 + 1] = str2[len_str2 + 1] = '\0';


    const int cols = len_str2 + 1;
    std::vector<int> mi_table(ALPHASIzE * cols);
    for (int i = 0; i < ALPHASIzE; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (j == 0)
            {
                mi_table[cols * i + j] = 0;
            }
            else if (AlphaBATES[i] == str2[j])
            {
                mi_table[cols * i + j] = j;
            }
            else
            {
                mi_table[cols * i + j] = mi_table[cols * i + (j - 1)];
            }
        }
    }


    {
        celerity::distr_queue queue;


        celerity::buffer<int, 2> mi_table_buf(mi_table.data(), cl::sycl::range<2>(ALPHASIzE, cols));
        celerity::buffer<unsigned char, 1> str2_buf(str2, cl::sycl::range<1>(len_str2 + 2));

        celerity::buffer<int, 1> row_pre_buf(cl::sycl::range<1>(len_str2 + 1));
        celerity::buffer<int, 1> row_curr_buf(cl::sycl::range<1>(len_str2 + 1));
        init_first_row<int>(queue, row_pre_buf);


        for (int i = 1; i <= len_str1; i++)
        {
            char chS1 = str1[i];
            int ci = 0;
            if (chS1 == AlphaBATES[1])
                ci = 1;
            else if (chS1 == AlphaBATES[2])
                ci = 2;
            else if (chS1 == AlphaBATES[3])
                ci = 3;
            if (i % 2 != 0)
                compute_ith_row(queue, row_pre_buf, row_curr_buf, mi_table_buf, str2_buf, str1[i], ci, i, cols);
            else
                compute_ith_row(queue, row_curr_buf, row_pre_buf, mi_table_buf, str2_buf, str1[i], ci, i, cols);

        }

        auto result_row = &row_curr_buf;

        if (len_str1 % 2 == 0)
        {
            result_row = &row_pre_buf;
        }

        int edit_dis = 0;
        auto range = cl::sycl::range<1>(cols);
        queue.submit(celerity::allow_by_ref, [&](celerity::handler& cgh) {
            auto result = result_row->get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::one_to_one<1>());
            cgh.host_task(range, [=, &edit_dis](celerity::partition<1> part) {

                auto sr = part.get_subrange();
                edit_dis = result[{sr.offset[0] + sr.range[0] - 1}];

                if (rank == numProc - 1)
                    printf("Edit distance = %d\n", edit_dis);
                });




            });


    }
    return EXIT_SUCCESS;
}
