//
//  AtomicEnum.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 24.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct AtomicEnum<T: RawRepresentable> where T.RawValue == Int {
    
    private var atomic: AtomicInt
    
    @inlinable init(value: T) {
        atomic = AtomicInt(value: value.rawValue)
    }
    
    @inlinable var value: T {
        get { T(rawValue: atomic.value)! }
        set { atomic.value = newValue.rawValue }
    }
    
    @inlinable mutating func update(_ newValue: T) -> T {
        T(rawValue: atomic.update(newValue.rawValue))!
    }
    
}
