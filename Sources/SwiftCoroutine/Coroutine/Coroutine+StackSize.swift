//
//  Coroutine+StackSize.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#if os(macOS)
import Darwin
#else
import Glibc
#endif

extension Coroutine {
    
    public struct StackSize {
        public let size: Int
    }
    
}

extension Coroutine.StackSize {
    
    public static let recommended = Coroutine.StackSize(size: Int(SIGSTKSZ))
    public static let minimal = Coroutine.StackSize(size: Int(MINSIGSTKSZ))
    
    public static func pages(_ number: Int) -> Coroutine.StackSize {
        .init(size: number * .pageSize)
    }
    
}

