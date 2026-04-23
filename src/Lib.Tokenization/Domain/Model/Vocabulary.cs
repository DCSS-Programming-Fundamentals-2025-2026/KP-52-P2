namespace  Lib.Tokenization.Domain.Model;

public class Vocabulary<T>
{
    private T[] _items;
    private int _count;

    public Vocabulary()
    {
        _items = new T[100];
        _items[0] = default;
        _count = 1;
    }

    public int GetId(T item)
    {
        for (int i = 0; i < _count; i++)
        {
            if (_items[i] != null && _items[i].Equals(item))
            {
                return i;
            }
        }	
        return 0;
    }

    public void Add(T item)
    {
        if (GetId(item) > 0)
        {
            return;
        }
        else if (_count == _items.Length)
        {
            T[] newItems = new T[_items.Length * 2];
            Array.Copy(_items, newItems, _count);
            _items = newItems;
        }
        _items[_count] = item;
        _count++;
    }

    public T GetItem(int id)
    {
        if (id < _count && id >= 0)
        {
            return _items[id];
        }
        return _items[0];
    }
    public int Count => _count;
}