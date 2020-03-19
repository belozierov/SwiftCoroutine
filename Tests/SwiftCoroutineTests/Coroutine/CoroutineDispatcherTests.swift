//
//  CoroutineDispatcherTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoroutineDispatcherTests: XCTestCase {
    
    func testMeasure() {
        let group = DispatchGroup()
        measure {
            for _ in 0..<10_000 {
                group.enter()
                CoroutineDispatcher.main.execute(group.leave)
            }
            group.wait()
        }
    }
    
    func testMeasure2() {
        let group = DispatchGroup()
        measure {
            for _ in 0..<10_000 {
                group.enter()
                CoroutineDispatcher.global.execute(group.leave)
            }
            group.wait()
        }
    }
    
   func testMeasure3() {
        let group = DispatchGroup()
        measure {
            for _ in 0..<10_000 {
                group.enter()
                CoroutineDispatcher.global.execute {
                    let coroutine = try! Coroutine.current()
                    try! Coroutine.suspend { try! coroutine.resume() }
                    group.leave()
                }
            }
            group.wait()
        }
    }
    
}
